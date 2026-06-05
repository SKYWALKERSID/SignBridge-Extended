import numpy as np
import json, joblib, os, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*60)
print("SignBridge — Random Forest Training")
print("="*60)

data = np.load(os.path.join(SCRIPT_DIR, 'landmarks_clean.npz'))
X, y = data['landmarks'], data['labels']

print(f"Samples:  {len(X):,}")
print(f"Classes:  {len(np.unique(y))}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

print(f"Train: {len(y_train):,}  Val: {len(y_val):,}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

TOTAL_TREES = 100
EVAL_EVERY  = 10

print(f"\nBuilding {TOTAL_TREES} trees (evaluating every {EVAL_EVERY})...\n")
print(f"{'Trees':>6} | {'Train Acc':>9} | {'Val Acc':>8} | {'Time':>6}")
print("-" * 40)

best_acc  = 0
best_n    = 0
all_start = time.time()

for n in range(EVAL_EVERY, TOTAL_TREES + 1, EVAL_EVERY):
    t = time.time()

    rf = RandomForestClassifier(
        n_estimators=n,
        n_jobs=2,
        class_weight='balanced',
        random_state=42,
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=1,
        warm_start=False,
    )
    rf.fit(X_train, y_train)

    val_acc   = accuracy_score(y_val,   rf.predict(X_val))
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    elapsed   = time.time() - t

    tag = ""
    if val_acc > best_acc:
        best_acc = val_acc
        best_n   = n
        tag      = " * best"
        joblib.dump(rf, os.path.join(SCRIPT_DIR, 'sign_model.pkl'))

    print(f"{n:>6} | {train_acc:>9.3f} | {val_acc:>8.3f} | {elapsed:>5.1f}s{tag}")

total_time = time.time() - all_start

print(f"\n{'='*60}")
print(f"  BEST VALIDATION ACCURACY: {best_acc*100:.2f}% at {best_n} trees")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"{'='*60}")

joblib.dump(scaler, os.path.join(SCRIPT_DIR, 'scaler.pkl'))
with open(os.path.join(SCRIPT_DIR, 'model_classes.json'), 'w') as f:
    json.dump(list(range(len(np.unique(y)))), f)

print(f"\nSaved:")
print(f"  sign_model.pkl  ({best_n} trees)")
print(f"  scaler.pkl")
print(f"  model_classes.json")

report = classification_report(
    y_val,
    joblib.load(os.path.join(SCRIPT_DIR, 'sign_model.pkl')).predict(X_val),
    output_dict=True)
print(f"\nMacro F1: {report['macro avg']['f1-score']:.3f}")
print(f"Macro P:  {report['macro avg']['precision']:.3f}")
print(f"Macro R:  {report['macro avg']['recall']:.3f}")