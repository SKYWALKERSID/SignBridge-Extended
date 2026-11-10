# SignBridge Extended

**SignBridge Extended** is a real-time Indian Sign Language (ISL) bidirectional translator (Sign-to-Text/Speech and Text/Speech-to-Sign) supporting **English** and **Hindi (Devanagari)**.

It pairs client-side **MediaPipe Hands** 3D skeletal tracking with a **Flask** inference backend featuring a 3-tier classification pipeline (**PyTorch BiLSTM with Attention** -> **Scikit-Learn Random Forest** -> **Built-in 3D Geometric Pose Classifier**), temporal vote smoothing, Web Speech synthesis/recognition, and a spring-physics volumetric hand avatar.

---

## Key Features

- **Live Dual-Hand Sign-to-Text & Voice (`Live Translator`):**
  - Tracks 21 3D hand landmarks (`63` coordinates per hand) for up to 2 hands simultaneously inside the browser.
  - Wrist-origin translation and palm-size scale normalization (`points[0]` to `points[9]`) for distance-independent recognition.
  - Rolling 6-frame temporal smoothing buffer (`>= 4` stable votes at `>= 70%` confidence) to prevent prediction chatter.
  - Integrated Web Speech API Text-to-Speech (TTS) vocalization and persistent session log saving.
- **Bilingual Text-to-Sign & Voice-to-Sign (`Text-to-Sign`):**
  - Supports **English** and **Hindi (Devanagari)** text input as well as live microphone Speech-to-Text (`SpeechRecognition`).
  - Renders sequential gestures on a 2.5D/3D procedural volumetric hand avatar driven by damped spring physics.
  - Automatic character-by-character fingerspelling fallback (`A–Z`, `0–9`, and Devanagari) for unmapped words.
- **Insights, Logs & Practice Arena (`Insights & Log`):**
  - Searchable local session transcript archive persisted in `localStorage`.
  - Real-time knuckle cosine alignment calculator comparing live webcam hand geometry against target calibration poses.
- **Accessibility First:**
  - Dual-theme Light/Dark academic palette (`Alt + T`), keyboard shortcuts (`Alt + A`), sensory voice feedback (`Alt + V`), high-contrast outlines, and dynamic font scaling.

---

## Project Structure

```text
SignBridge-Extended/
├── CODE FILES/
│   ├── app.py                 # Flask backend server & 3-tier inference engine
│   ├── index.html             # React 18 + Tailwind CSS single-page frontend
│   ├── model_classes.json     # 251 ISL class definitions (0-9, A-Z, words, sentences)
│   ├── train.py               # Full dataset landmark extractor & 300-tree RF trainer
│   ├── train_rf.py            # Random Forest hyperparameter sweep script
│   └── train_lstm.py          # PyTorch Bidirectional LSTM + Attention trainer
├── Document and start files/
│   ├── Start-SignBridge.bat   # One-click Windows batch launcher
│   ├── Start-SignBridge.ps1   # PowerShell setup & server launcher script
│   └── CITATION FOR INCLUDE.txt # INCLUDE dataset citation
├── MD FILES/
│   └── ARCHITECTURE.md        # Technical architecture & API documentation
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview & quickstart guide
```

---

## Quick Start

### Option 1: One-Click Windows Launcher
Double-click [`Document and start files/Start-SignBridge.bat`](Document%20and%20start%20files/Start-SignBridge.bat) or run via PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File ".\Document and start files\Start-SignBridge.ps1"
```
The launcher verifies required Python packages, frees port `8000` if busy, starts the Flask backend in `CODE FILES/`, and opens `http://127.0.0.1:8000` in your browser.

### Option 2: Manual Startup
1. Install runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the backend server:
   ```bash
   python "CODE FILES/app.py"
   ```
3. Open **http://127.0.0.1:8000** in Google Chrome or Microsoft Edge.

---

## Inference Engine Modes

`CODE FILES/app.py` automatically selects the highest-priority classifier available in `CODE FILES/`:

1. **BiLSTM Mode (`engine: "lstm"`):** Active when `sign_model_lstm.pt`, `scaler_lstm.pkl`, and `model_classes_lstm.json` are placed in `CODE FILES/` (requires `torch`).
2. **Random Forest Mode (`engine: "random_forest"`):** Active when `sign_model.pkl` and `scaler.pkl` are placed in `CODE FILES/`.
3. **Geometric 3D Fallback Mode (`engine: "geometric_fallback"`):** Active out-of-the-box when external `.pt`/`.pkl` model weights are absent, enabling immediate live recognition for core ISL signs (`hello`, `thanks`, `yes`, `no`, `iloveyou`, `you`, `GOOD`, `FINE`, numbers, and alphabet poses).

### Training Custom Model Weights
To train `sign_model.pkl` and `scaler.pkl` from the raw ISL datasets (`DATA/original_images`, `DATA/Dataset`, `DATA/Indian`, and `DATA/ISL_CSLRT_Corpus`):
```bash
pip install opencv-python mediapipe openpyxl
python "CODE FILES/train.py"
```
If you already have extracted landmark caches (`landmarks_clean.npz` or `landmarks_cache.npz` inside `CODE FILES/`):
```bash
python "CODE FILES/train_rf.py"
```
To train the PyTorch Bidirectional LSTM with Attention (`sign_model_lstm.pt`, `scaler_lstm.pkl`, and `model_classes_lstm.json`):
```bash
pip install torch
python "CODE FILES/train_lstm.py"
```

---

## API Reference

### `GET /api/v1/health`
Returns server status and the active classification engine.
```json
{
  "status": "ok",
  "model_loaded": false,
  "engine": "geometric_fallback",
  "classes_count": 251,
  "version": "1.1.0"
}
```

### `GET /api/v1/classes`
Returns the loaded ISL class list (`251` classes by default from `model_classes.json`).

### `POST /api/v1/infer/landmarks`
Classifies a 63-float array (`21` 3D hand joints `[x, y, z]`) and applies temporal vote smoothing.
- **Request:**
  ```json
  {
    "landmarks": [0.51, 0.72, 0.0, "... (63 floats total)"],
    "session_id": "signbridge-portal-session-h1"
  }
  ```
- **Response:**
  ```json
  {
    "prediction": "hello",
    "confidence": 0.91,
    "top_k": [
      {"label": "hello", "confidence": 0.91},
      {"label": "5", "confidence": 0.06},
      {"label": "STOP", "confidence": 0.03}
    ],
    "smoothed_prediction": "hello",
    "commit": true,
    "ts": 1731240000000
  }
  ```

### `POST /api/v1/text-to-sign`
Maps English (`en`) or Hindi (`hi`) text into sequential sign animation tokens and identifies unmapped words for fingerspelling fallback.
- **Request:**
  ```json
  {
    "text": "नमस्ते धन्यवाद हाँ",
    "language": "hi"
  }
  ```
- **Response:**
  ```json
  {
    "normalized_text": "नमस्ते धन्यवाद हाँ",
    "tokens": ["नमस्ते", "धन्यवाद", "हाँ"],
    "sequence": [
      {"token": "नमस्ते", "animation_id": "joined-palms-bow", "duration_ms": 700},
      {"token": "धन्यवाद", "animation_id": "chin-forward-thank", "duration_ms": 700},
      {"token": "हाँ", "animation_id": "fist-nod", "duration_ms": 700}
    ],
    "fallback_tokens": []
  }
  ```

---

## Team & Citation

**Project Team:**
- **Siddhesh Tiwari** — Ideation, Web Development & MediaPipe Integration
- **Muhammed Habeeb** — Core ML Code Development & Dataset Curation
- **Tanishq Mehtab** — Web Testing, Accessibility & Debugging
- **Varun Yadav** — Documentation, Architecture & Change Logs

**Project Guides:** Ms. Vandana Patel, Mahesh Bhagwani

**Dataset Citation:**
If you use the INCLUDE dataset, please cite:
> Sridhar, A., Ganesan, R. G., Kumar, P., & Khapra, M. (2020). *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*. Proceedings of the 28th ACM International Conference on Multimedia (MM '20), 1366–1375. https://doi.org/10.1145/3394171.3413528

