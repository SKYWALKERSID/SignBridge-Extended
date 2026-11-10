import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("SignBridge — Random Forest Training")
print("=" * 60)

clean_npz = os.path.join(SCRIPT_DIR, "landmarks_clean.npz")
cache_npz = os.path.join(SCRIPT_DIR, "landmarks_cache.npz")

if os.path.exists(clean_npz):
    npz_path = clean_npz
elif os.path.exists(cache_npz):
    npz_path = cache_npz
else:
    print(f"[ERROR] Neither {clean_npz} nor {cache_npz} found.")
    print("Run `python train.py` first to extract landmarks from the dataset.")
    sys.exit(1)

print(f"Loading dataset cache: {npz_path}")
data = np.load(npz_path, allow_pickle=True)
if "landmarks" in data and "labels" in data:
    X, y = data["landmarks"], data["labels"]
elif "X" in data and "y" in data:
    X, y = data["X"], data["y"]
else:
    print(f"[ERROR] Unexpected keys in {npz_path}: {list(data.keys())}")
    sys.exit(1)

unique_classes = sorted(list(np.unique(y)))
print(f"Samples:  {len(X):,}")
print(f"Classes:  {len(unique_classes)}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"Train: {len(y_train):,}  Val: {len(y_val):,}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

TOTAL_TREES = 100
EVAL_EVERY = 10

print(f"\nBuilding {TOTAL_TREES} trees (evaluating every {EVAL_EVERY})...\n")
print(f"{'Trees':>6} | {'Train Acc':>9} | {'Val Acc':>8} | {'Time':>6}")
print("-" * 40)

best_acc = 0.0
best_n = 0
all_start = time.time()

model_out_path = os.path.join(SCRIPT_DIR, "sign_model.pkl")
scaler_out_path = os.path.join(SCRIPT_DIR, "scaler.pkl")
classes_out_path = os.path.join(SCRIPT_DIR, "model_classes.json")

for n in range(EVAL_EVERY, TOTAL_TREES + 1, EVAL_EVERY):
    t = time.time()

    rf = RandomForestClassifier(
        n_estimators=n,
        n_jobs=2,
        class_weight="balanced",
        random_state=42,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        warm_start=False,
    )
    rf.fit(X_train, y_train)

    val_acc = accuracy_score(y_val, rf.predict(X_val))
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    elapsed = time.time() - t

    tag = ""
    if val_acc > best_acc:
        best_acc = val_acc
        best_n = n
        tag = " * best"
        joblib.dump(rf, model_out_path)

    print(f"{n:>6} | {train_acc:>9.3f} | {val_acc:>8.3f} | {elapsed:>5.1f}s{tag}")

total_time = time.time() - all_start

print(f"\n{'=' * 60}")
print(f"  BEST VALIDATION ACCURACY: {best_acc * 100:.2f}% at {best_n} trees")
print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
print(f"{'=' * 60}")

joblib.dump(scaler, scaler_out_path)

# Preserve string class names instead of overwriting with integer indices
existing_classes = None
if os.path.exists(classes_out_path):
    try:
        with open(classes_out_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list) and all(isinstance(c, str) for c in loaded):
                existing_classes = loaded
    except Exception:
        pass

if all(isinstance(c, (int, np.integer)) for c in unique_classes) and existing_classes and len(existing_classes) == len(unique_classes):
    classes_to_save = existing_classes
else:
    classes_to_save = [str(c) for c in unique_classes]

with open(classes_out_path, "w", encoding="utf-8") as f:
    json.dump(classes_to_save, f, indent=2)

print("\nSaved:")
print(f"  sign_model.pkl     ({best_n} trees)")
print("  scaler.pkl")
print(f"  model_classes.json ({len(classes_to_save)} classes)")

report = classification_report(
    y_val,
    joblib.load(model_out_path).predict(X_val),
    output_dict=True,
)
print(f"\nMacro F1: {report['macro avg']['f1-score']:.3f}")
print(f"Macro P:  {report['macro avg']['precision']:.3f}")
print(f"Macro R:  {report['macro avg']['recall']:.3f}")
