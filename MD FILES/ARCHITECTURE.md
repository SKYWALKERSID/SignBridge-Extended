# SignBridge Extended — System Architecture

## 1. Overview

**SignBridge Extended** is a real-time Indian Sign Language (ISL) bidirectional translation platform supporting:
- **Live Sign-to-Text & Voice (TTS):** Dual-hand 3D skeletal tracking in the browser via MediaPipe Hands, paired with a Flask inference backend supporting a 3-tier classification hierarchy (PyTorch Bidirectional LSTM with Attention -> Scikit-Learn Random Forest -> 3D Geometric Pose Classifier).
- **Text & Speech-to-Sign Animation (STT):** Bilingual English and Hindi (Devanagari) tokenization with a damped spring-physics 2.5D/3D procedural hand avatar and character-by-character fingerspelling fallback.

---

## 2. Repository Structure

```text
SignBridge-Extended/
├── CODE FILES/
│   ├── app.py                 # Flask API server & 3-tier inference engine (BiLSTM / RF / Geometric)
│   ├── index.html             # Single-file React 18 + Tailwind CSS frontend application
│   ├── model_classes.json     # 251 ISL class labels (0-9, A-Z, words, and ISL-CSLRT sentences)
│   ├── train.py               # Full dataset landmark extraction & Random Forest (300 trees) trainer
│   ├── train_rf.py            # Fast Random Forest hyperparameter sweep trainer (from cached .npz)
│   └── train_lstm.py          # PyTorch Bidirectional LSTM + Attention trainer
├── Document and start files/
│   ├── Start-SignBridge.bat   # Windows one-click batch launcher
│   ├── Start-SignBridge.ps1   # PowerShell environment check, port cleanup & server starter
│   └── CITATION FOR INCLUDE.txt # Citation for the INCLUDE dataset
├── MD FILES/
│   └── ARCHITECTURE.md        # System architecture documentation
├── requirements.txt           # Python runtime & training dependencies
└── README.md                  # Project documentation and setup guide
```

---

## 3. Runtime Pipeline

### 3.1 Sign-to-Text Pipeline
1. **Browser Capture (`index.html`):**
   - `navigator.mediaDevices.getUserMedia` streams 640x480 webcam frames.
   - `@mediapipe/hands` extracts 21 3D joints (`[x, y, z]` -> 63 floats) for up to 2 hands (`H1` and `H2`) in real time.
2. **Spatial Normalization (`normalize_landmarks` in `app.py`):**
   - Translates all 21 joints relative to the wrist origin (`points[0]`) for translation invariance.
   - Divides coordinates by the Euclidean distance between the wrist (`points[0]`) and middle finger MCP knuckle (`points[9]`) for scale/distance invariance.
   - Clips outliers to `[-3.0, 3.0]`.
3. **3-Tier Classification Engine (`app.py`):**
   - **Tier 1 — Bidirectional LSTM (`SignLanguageLSTM`):** Used when `sign_model_lstm.pt`, `scaler_lstm.pkl`, and `model_classes_lstm.json` are present. Buffers up to 30 frames per session and applies attention pooling over a 2-layer BiLSTM (`hidden_size=256`).
   - **Tier 2 — Random Forest Classifier:** Used when `sign_model.pkl` and `scaler.pkl` are present. Scales the 63-d normalized vector via `StandardScaler` and predicts class probabilities across 251 ISL classes.
   - **Tier 3 — Built-in 3D Geometric Pose Classifier (`classify_landmarks_geometric`):** Zero-dependency fallback when external `.pt`/`.pkl` model weights are absent. Evaluates 3D finger extension ratios, thumb-index pinch distances, and inter-finger spread to classify core ISL gestures (`hello`, `thanks`, `yes`, `no`, `iloveyou`, `you`, `GOOD`, `FINE`, numbers, and letters).
4. **Temporal Smoothing & Commit Gating:**
   - Maintains a per-session rolling window (`deque(maxlen=6)`).
   - Emits `commit = True` when the mode prediction appears at least `4` times in the 6-frame window with confidence `>= 0.70`.
   - Committed tokens are appended to the live transcript and spoken aloud via the browser's `speechSynthesis` API.

### 3.2 Text-to-Sign & Speech-to-Sign Pipeline
1. **Input Capture:** Accepts typed text or live voice input via Web Speech Recognition (`SpeechRecognition` / `webkitSpeechRecognition`) in English (`en-US`) or Hindi (`hi-IN`).
2. **Tokenization (`POST /api/v1/text-to-sign`):**
   - Strips surrounding punctuation while preserving Latin and Devanagari characters.
   - Maps known English and Hindi vocabulary to canonical gesture keys (`open-palm-wave`, `joined-palms-bow`, `chin-forward-thank`, `fist-nod`, `index-middle-tap`, `index-self`, `cross-arms-heart`, `point-forward`) and identifies unmapped words as `fallback_tokens`.
3. **Procedural Avatar Rendering (`drawVolumetricHand` & `getPoseCoordinates`):**
   - Animates a 21-joint volumetric hand on HTML5 Canvas using damped spring physics (`k = 0.30`, `d = 0.52`).
   - Unmapped tokens automatically fall back to sequential character-by-character fingerspelling poses.

---

## 4. REST API Endpoints (`app.py`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the main `index.html` React application |
| `GET` | `/api/v1/health` | Returns server status, active inference engine (`lstm`, `random_forest`, or `geometric_fallback`), and class count |
| `GET` | `/api/v1/classes` | Returns the active list of ISL classes and total count |
| `POST` | `/api/v1/infer/landmarks` | Accepts `{ "landmarks": [63 floats], "session_id": "..." }` and returns `prediction`, `confidence`, `top_k`, `smoothed_prediction`, and `commit` |
| `POST` | `/api/v1/text-to-sign` | Accepts `{ "text": "...", "language": "en" \| "hi" }` and returns normalized tokens, mapped gesture sequence, and fallback tokens |
