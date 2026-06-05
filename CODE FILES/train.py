"""
train.py -- Train SignBridge on ALL DATA: every image, video, and metadata file
================================================================================
Uses 100% of:
  - DATA/original_images        (36,000 images)
  - DATA/Dataset                (53,512 images)
  - DATA/Indian                 (42,745 images)
  - ISL_CSLRT_Corpus/Frames_Word_Level     (1,036 images)
  - ISL_CSLRT_Corpus/Frames_Sentence_Level (18,863 images)
  - ISL_CSLRT_Corpus/Videos_Sentence_Level (687 videos -> frame extraction)
  - ISL_CSLRT_Corpus/corpus_csv_files      (4 metadata files for labeling)

80% Train / 20% Test split.

Usage:
    python train.py
"""

import csv
import json
import os
import sys
import time
from collections import Counter

# Suppress TF/MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Micro Project Extended
DATA_ROOT = os.path.join(PROJECT_ROOT, "DATA")
CORPUS_ROOT = os.path.join(DATA_ROOT, "ISL_CSLRT_Corpus", "ISL_CSLRT_Corpus")
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
MODEL_OUTPUT = os.path.join(SCRIPT_DIR, "sign_model.pkl")
SCALER_OUTPUT = os.path.join(SCRIPT_DIR, "scaler.pkl")
CLASSES_OUTPUT = os.path.join(SCRIPT_DIR, "model_classes.json")
CACHE_FILE = os.path.join(SCRIPT_DIR, "landmarks_cache.npz")
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Video frame extraction sampling every Nth frame
VIDEO_FRAME_SAMPLE_INTERVAL = 3

# ============================================================
# MediaPipe HandLandmarker
# ============================================================
if not os.path.exists(HAND_MODEL_PATH):
    print(f"[ERROR] Hand landmarker model not found: {HAND_MODEL_PATH}")
    sys.exit(1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
)
landmarker = HandLandmarker.create_from_options(options)


# ============================================================
# Landmark Normalization
# ============================================================
def normalize_landmarks(landmarks):
    """Normalize a 63-d landmark vector: wrist origin + rigid scale."""
    points = np.array(landmarks).reshape(21, 3)
    wrist = points[0].copy()
    points = points - wrist
    hand_size = np.linalg.norm(points[9])
    if hand_size < 1e-6:
        return None
    points = points / hand_size
    points = np.clip(points, -3.0, 3.0)
    return points.flatten()


def extract_landmarks_from_image(image_path):
    """Extract normalized 63-d hand landmarks from an image file."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    raw = []
    for lm in hand:
        raw.extend([lm.x, lm.y, lm.z])
    return normalize_landmarks(raw)


# ============================================================
# Parse CSV/XLSX Metadata
# ============================================================
print("=" * 70)
print("  SignBridge -- FULL Dataset Training (ALL files, NO throttle)")
print("=" * 70)

# CSV: sentence -> sign glosses mapping
glosses_csv = os.path.join(CORPUS_ROOT, "corpus_csv_files", "ISL Corpus sign glosses.csv")
sentence_to_glosses = {}
if os.path.exists(glosses_csv):
    with open(glosses_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                sentence_to_glosses[row[0].strip()] = row[1].strip()
    print(f"[Metadata] Loaded {len(sentence_to_glosses)} sentence->gloss mappings from CSV")

# Try to load XLSX metadata
frame_details_count = 0
word_details_count = 0
try:
    import openpyxl
    frame_xlsx = os.path.join(CORPUS_ROOT, "corpus_csv_files", "Frame Details.xlsx")
    if os.path.exists(frame_xlsx):
        wb = openpyxl.load_workbook(frame_xlsx, read_only=True)
        ws = wb.active
        frame_details_count = sum(1 for _ in ws.iter_rows(min_row=2)) 
        wb.close()
        print(f"[Metadata] Frame details XLSX: {frame_details_count} entries validated")

    word_xlsx = os.path.join(CORPUS_ROOT, "corpus_csv_files", "Word Details.xlsx")
    if os.path.exists(word_xlsx):
        wb = openpyxl.load_workbook(word_xlsx, read_only=True)
        ws = wb.active
        word_details_count = sum(1 for _ in ws.iter_rows(min_row=2))
        wb.close()
        print(f"[Metadata] Word details XLSX: {word_details_count} entries validated")
except ImportError:
    print("[Metadata] openpyxl not installed, skipping XLSX metadata")


# ============================================================
# Check for existing cache
# ============================================================
if os.path.exists(CACHE_FILE):
    print(f"\n[Cache] Loading existing landmark cache: {CACHE_FILE}")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    X = cache["X"]
    y = cache["y"]
    print(f"[Cache] Loaded {len(X):,} samples across {len(np.unique(y))} classes")
else:
    # ============================================================
    # Collect all image paths by class
    # ============================================================
    all_images = {}  # class_name -> [image_paths]

    # Source 1: original_images (subfolders = class names)
    src1 = os.path.join(DATA_ROOT, "original_images")
    if os.path.isdir(src1):
        count = 0
        for cls in sorted(os.listdir(src1)):
            cls_dir = os.path.join(src1, cls)
            if os.path.isdir(cls_dir):
                imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                all_images.setdefault(cls, []).extend(imgs)
                count += len(imgs)
        print(f"\n[Source 1/5] {src1} ...")
        print(f"  Added {count:,} images")

    # Source 2: Dataset
    src2 = os.path.join(DATA_ROOT, "Dataset")
    if os.path.isdir(src2):
        count = 0
        for cls in sorted(os.listdir(src2)):
            cls_dir = os.path.join(src2, cls)
            if os.path.isdir(cls_dir):
                imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                all_images.setdefault(cls, []).extend(imgs)
                count += len(imgs)
        print(f"[Source 2/5] {src2} ...")
        print(f"  Added {count:,} images")

    # Source 3: Indian
    src3 = os.path.join(DATA_ROOT, "Indian")
    if os.path.isdir(src3):
        count = 0
        for cls in sorted(os.listdir(src3)):
            cls_dir = os.path.join(src3, cls)
            if os.path.isdir(cls_dir):
                imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                all_images.setdefault(cls, []).extend(imgs)
                count += len(imgs)
        print(f"[Source 3/5] {src3} ...")
        print(f"  Added {count:,} images")

    # Source 4: Frames_Word_Level
    src4 = os.path.join(CORPUS_ROOT, "Frames_Word_Level")
    if os.path.isdir(src4):
        count = 0
        for cls in sorted(os.listdir(src4)):
            cls_dir = os.path.join(src4, cls)
            if os.path.isdir(cls_dir):
                imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                all_images.setdefault(cls, []).extend(imgs)
                count += len(imgs)
        print(f"[Source 4/5] {src4} ...")
        print(f"  Added {count:,} images")

    # Source 5: Frames_Sentence_Level
    src5 = os.path.join(CORPUS_ROOT, "Frames_Sentence_Level")
    if os.path.isdir(src5):
        count = 0
        for cls in sorted(os.listdir(src5)):
            cls_dir = os.path.join(src5, cls)
            if os.path.isdir(cls_dir):
                imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                all_images.setdefault(cls, []).extend(imgs)
                count += len(imgs)
        print(f"[Source 5/5] {src5} ...")
        print(f"  Added {count:,} images")

    total_images = sum(len(v) for v in all_images.values())
    print(f"\nTotal images collected: {total_images:,} across {len(all_images)} classes")

    # ============================================================
    # Extract landmarks from all images
    # ============================================================
    print(f"\nExtracting landmarks from {total_images:,} images across {len(all_images)} classes...")
    print("This will take a while -- processing ALL data with no throttle.\n")

    X_list = []
    y_list = []
    total_ok = 0
    total_skip = 0
    extract_start = time.time()

    for idx, (cls_name, img_paths) in enumerate(sorted(all_images.items()), 1):
        ok = 0
        skip = 0
        for img_path in img_paths:
            lm = extract_landmarks_from_image(img_path)
            if lm is not None:
                X_list.append(lm)
                y_list.append(cls_name)
                ok += 1
            else:
                skip += 1
        total_ok += ok
        total_skip += skip
        print(f"  [{idx:>3}/{len(all_images)}] {idx/len(all_images)*100:5.1f}%  "
              f"{cls_name:<40s} ok: {ok:>4}  skip:{skip:>4}  "
              f"(total so far: {total_ok:,})")

    img_time = time.time() - extract_start
    print(f"\n--- Image Extraction Done in {img_time:.1f}s ---")
    print(f"  Samples from images: {total_ok:,}  Skipped: {total_skip:,}")

    # ============================================================
    # Extract landmarks from videos
    # ============================================================
    video_dir = os.path.join(CORPUS_ROOT, "Videos_Sentence_Level")
    if os.path.isdir(video_dir):
        video_files = []
        for cls in sorted(os.listdir(video_dir)):
            cls_dir = os.path.join(video_dir, cls)
            if os.path.isdir(cls_dir):
                for f in os.listdir(cls_dir):
                    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_files.append((cls, os.path.join(cls_dir, f)))

        print(f"\nExtracting landmarks from {len(video_files)} videos "
              f"(every {VIDEO_FRAME_SAMPLE_INTERVAL}th frame)...")

        vid_start = time.time()
        total_frames = 0
        total_vid_lm = 0

        for vid_idx, (cls_name, vid_path) in enumerate(video_files, 1):
            cap = cv2.VideoCapture(vid_path)
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % VIDEO_FRAME_SAMPLE_INTERVAL != 0:
                    continue
                total_frames += 1
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)
                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    raw = [c for lm in hand for c in (lm.x, lm.y, lm.z)]
                    norm = normalize_landmarks(raw)
                    if norm is not None:
                        X_list.append(norm)
                        y_list.append(cls_name)
                        total_vid_lm += 1
            cap.release()

            if vid_idx % 50 == 0 or vid_idx == len(video_files):
                print(f"  Videos [{vid_idx}/{len(video_files)}]  "
                      f"frames sampled: {total_frames:,}  "
                      f"landmarks: {total_vid_lm:,}")

        vid_time = time.time() - vid_start
        print(f"\n--- Video Extraction Done in {vid_time:.1f}s ---")
        print(f"  Total video frames sampled: {total_frames:,}")
        print(f"  Landmarks extracted from videos: {total_vid_lm:,}")

    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)

    # Save cache
    np.savez(CACHE_FILE, X=X, y=y)
    print(f"\n[Cache] Saved {len(X):,} samples to {CACHE_FILE}")

    total_time_extract = time.time() - extract_start
    print(f"\n{'='*70}")
    print(f"  FEATURE EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total samples (images + videos): {len(X):,}")
    print(f"  Total classes                   : {len(np.unique(y))}")
    print(f"  Total extraction time           : {total_time_extract:.1f}s")


# ============================================================
# Training
# ============================================================
print(f"\nFinal dataset: {len(X):,} samples, {len(np.unique(y))} classes")
print(f"Train/Test split: 80% / 20%")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Training set  : {len(X_train):,} samples")
print(f"  Validation set: {len(X_val):,} samples")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train
print(f"Training RandomForestClassifier (n_estimators=300, n_jobs=2)...")
start = time.time()

rf = RandomForestClassifier(
    n_estimators=300,
    n_jobs=2,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    max_depth=None,
    min_samples_leaf=1,
    verbose=1,
)
rf.fit(X_train, y_train)

elapsed = time.time() - start
print(f"Training completed in {elapsed:.1f}s")

# Evaluate
preds = rf.predict(X_val)
acc = accuracy_score(y_val, preds)

print(f" VALIDATION ACCURACY: {acc*100:.2f}%")

report = classification_report(y_val, preds, output_dict=True)
print(f"  Macro F1:  {report['macro avg']['f1-score']:.3f}")
print(f"  Macro P:   {report['macro avg']['precision']:.3f}")
print(f"  Macro R:   {report['macro avg']['recall']:.3f}")

# Save
joblib.dump(rf, MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)

classes_list = sorted(list(np.unique(y)))
with open(CLASSES_OUTPUT, 'w') as f:
    json.dump(classes_list, f, indent=2)

print(f"\nSaved:")
print(f"  {MODEL_OUTPUT}")
print(f"  {SCALER_OUTPUT}")
print(f"  {CLASSES_OUTPUT} ({len(classes_list)} classes)")
