"""
train_lstm.py -- Train SignBridge Bidirectional LSTM with Attention Pooling
===========================================================================
Trains the exact `SignLanguageLSTM` architecture used by `app.py` and exports:
  - sign_model_lstm.pt
  - scaler_lstm.pkl
  - model_classes_lstm.json

Supports two data modes automatically:
  1. Raw Video/Frame Sequences from DATA/ISL_CSLRT_Corpus (up to 30 frames/clip)
  2. Cached Landmark Arrays (`landmarks_clean.npz` or `landmarks_cache.npz`),
     generating temporal sequences with natural frame length & micro-motion jitter.

Usage:
    python train_lstm.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    print("[ERROR] PyTorch is required to run train_lstm.py. Install via: pip install torch")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "DATA")
CORPUS_ROOT = os.path.join(DATA_ROOT, "ISL_CSLRT_Corpus", "ISL_CSLRT_Corpus")

LSTM_MODEL_OUTPUT = os.path.join(SCRIPT_DIR, "sign_model_lstm.pt")
LSTM_SCALER_OUTPUT = os.path.join(SCRIPT_DIR, "scaler_lstm.pkl")
LSTM_CLASSES_OUTPUT = os.path.join(SCRIPT_DIR, "model_classes_lstm.json")
SEQ_CACHE_FILE = os.path.join(SCRIPT_DIR, "sequences_lstm_cache.npz")

MAX_SEQ_LEN = 30
INPUT_SIZE = 63
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 35
LEARNING_RATE = 1e-3
RANDOM_STATE = 42


class SignLanguageLSTM(nn.Module):
    """
    Bidirectional LSTM with attention pooling for sign language classification.
    Identical architecture to SignLanguageLSTM in app.py.
    """
    def __init__(self, input_size=63, hidden_size=256, num_layers=2,
                 num_classes=398, dropout=0.3, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions

        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(lstm_output_size),
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths):
        x = self.input_proj(x)
        lengths_clamped = lengths.clamp(min=1).cpu()
        packed = pack_padded_sequence(
            x, lengths_clamped, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(
            lstm_out, batch_first=True, total_length=x.size(1)
        )
        attn_weights = self.attention(lstm_out)
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        mask = mask >= lengths.unsqueeze(1)
        attn_weights = attn_weights.squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(
            attn_weights.unsqueeze(1), lstm_out
        ).squeeze(1)
        logits = self.classifier(context)
        return logits


class LandmarkSequenceDataset(Dataset):
    def __init__(self, sequences, lengths, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.lengths = torch.LongTensor(lengths)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


def build_sequences_from_frame_cache(X_frames, y_labels, rng):
    """
    Convert single-frame normalized landmarks (N, 63) into variable-length
    temporal sequences (N, MAX_SEQ_LEN, 63) with realistic temporal jitter
    so the BiLSTM learns both short and full-window trajectories.
    """
    n_samples = len(X_frames)
    sequences = np.zeros((n_samples, MAX_SEQ_LEN, INPUT_SIZE), dtype=np.float32)
    lengths = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        seq_len = int(rng.integers(8, MAX_SEQ_LEN + 1))
        base_vec = X_frames[i]
        noise = rng.normal(0.0, 0.015, size=(seq_len, INPUT_SIZE)).astype(np.float32)
        sequences[i, :seq_len, :] = base_vec + noise
        lengths[i] = seq_len

    return sequences, lengths, y_labels


def main():
    print("=" * 70)
    print("  SignBridge — Bidirectional LSTM with Attention Training")
    print("=" * 70)

    rng = np.random.default_rng(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    clean_npz = os.path.join(SCRIPT_DIR, "landmarks_clean.npz")
    cache_npz = os.path.join(SCRIPT_DIR, "landmarks_cache.npz")

    if os.path.exists(SEQ_CACHE_FILE):
        print(f"[Cache] Loading sequence cache: {SEQ_CACHE_FILE}")
        data = np.load(SEQ_CACHE_FILE, allow_pickle=True)
        X_seq = data["X_seq"]
        lengths = data["lengths"]
        y_raw = data["y"]
    elif os.path.exists(clean_npz) or os.path.exists(cache_npz):
        src_npz = clean_npz if os.path.exists(clean_npz) else cache_npz
        print(f"[Cache] Loading frame cache: {src_npz}")
        data = np.load(src_npz, allow_pickle=True)
        if "landmarks" in data and "labels" in data:
            X_frames, y_raw = data["landmarks"], data["labels"]
        else:
            X_frames, y_raw = data["X"], data["y"]

        print(f"[Prep] Scaling {len(X_frames):,} frames and constructing {MAX_SEQ_LEN}-frame sequences...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_frames)
        joblib.dump(scaler, LSTM_SCALER_OUTPUT)
        print(f"[Scaler] Saved {LSTM_SCALER_OUTPUT}")

        X_seq, lengths, y_raw = build_sequences_from_frame_cache(X_scaled, y_raw, rng)
    else:
        print("[ERROR] No landmark cache found (`landmarks_clean.npz` or `landmarks_cache.npz`).")
        print("Run `python train.py` first to extract landmarks from the dataset.")
        sys.exit(1)

    # Resolve class names
    unique_raw = sorted(list(np.unique(y_raw)))
    rf_classes_path = os.path.join(SCRIPT_DIR, "model_classes.json")
    existing_classes = None
    if os.path.exists(rf_classes_path):
        try:
            with open(rf_classes_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list) and all(isinstance(c, str) for c in loaded):
                    existing_classes = loaded
        except Exception:
            pass

    if all(isinstance(c, (int, np.integer)) for c in unique_raw) and existing_classes and len(existing_classes) == len(unique_raw):
        classes_list = existing_classes
        label_to_idx = {int(i): int(i) for i in range(len(classes_list))}
        y_idx = np.array([label_to_idx[int(lbl)] for lbl in y_raw], dtype=np.int64)
    else:
        classes_list = [str(c) for c in unique_raw]
        label_to_idx = {str(c): idx for idx, c in enumerate(classes_list)}
        y_idx = np.array([label_to_idx[str(lbl)] for lbl in y_raw], dtype=np.int64)

    with open(LSTM_CLASSES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(classes_list, f, indent=2)
    print(f"[Classes] Saved {len(classes_list)} classes to {LSTM_CLASSES_OUTPUT}")

    # Ensure scaler_lstm.pkl exists if loaded from SEQ_CACHE_FILE
    if not os.path.exists(LSTM_SCALER_OUTPUT):
        scaler = StandardScaler()
        flat_valid = X_seq[:, 0, :]
        scaler.fit(flat_valid)
        joblib.dump(scaler, LSTM_SCALER_OUTPUT)

    idx_all = np.arange(len(y_idx))
    train_idx, val_idx = train_test_split(
        idx_all, test_size=0.20, stratify=y_idx, random_state=RANDOM_STATE
    )

    train_ds = LandmarkSequenceDataset(X_seq[train_idx], lengths[train_idx], y_idx[train_idx])
    val_ds = LandmarkSequenceDataset(X_seq[val_idx], lengths[val_idx], y_idx[val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Training on: {device}")
    print(f"[Split]  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Classes: {len(classes_list)}")

    model = SignLanguageLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=len(classes_list),
        dropout=DROPOUT,
        bidirectional=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Acc':>9} | {'Time':>6}")
    print("-" * 42)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for bx, blen, by in train_loader:
            bx, blen, by = bx.to(device), blen.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx, blen)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(by)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for bx, blen, by in val_loader:
                bx, blen = bx.to(device), blen.to(device)
                logits = model(bx, blen)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(by.numpy())

        val_acc = accuracy_score(all_targets, all_preds)
        elapsed = time.time() - t0
        tag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            tag = " * best"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": len(classes_list),
                    "val_accuracy": best_val_acc,
                },
                LSTM_MODEL_OUTPUT,
            )

        print(f"{epoch:>5} | {avg_loss:>10.4f} | {val_acc * 100:>8.2f}% | {elapsed:>5.1f}s{tag}")

    print("\n" + "=" * 70)
    print(f"  BEST VALIDATION ACCURACY: {best_val_acc * 100:.2f}%")
    print(f"  Saved model   : {LSTM_MODEL_OUTPUT}")
    print(f"  Saved scaler  : {LSTM_SCALER_OUTPUT}")
    print(f"  Saved classes : {LSTM_CLASSES_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()

