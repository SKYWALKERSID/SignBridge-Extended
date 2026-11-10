# SignBridge Extended

[![License: MIT](https://img.shields.io/badge/License-MIT-435b41.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MediaPipe Hands](https://img.shields.io/badge/MediaPipe-3D%20Hands-orange.svg)](https://developers.google.com/mediapipe)
[![ISL Classes](https://img.shields.io/badge/Vocabulary-251%20ISL%20Classes-success.svg)](CODE%20FILES/model_classes.json)

**SignBridge Extended** is a real-time, bidirectional **Indian Sign Language (ISL)** translation suite that converts live hand gestures into spoken and written text, and translates **English** and **Hindi (Devanagari)** text or speech back into animated 3D hand signs.

All skeletal tracking, spatial normalization, and neural classification run locally on your machine with zero cloud dependencies.

---

## Highlights

- **Real-Time Dual-Hand Sign-to-Text (`Live Translator`)**
  - Captures 21 3D skeletal hand joints (`63` coordinates per hand) for up to 2 hands simultaneously using MediaPipe Hands.
  - Performs wrist-origin translation and palm-scale normalization (`points[0]` to `points[9]`) so recognition stays accurate regardless of hand size or distance from the camera.
  - Uses a rolling 6-frame temporal voting buffer (`>= 4` matching frames at `>= 70%` confidence) to eliminate prediction flicker and auto-speak committed phrases via Web Speech TTS.
- **Bilingual Text & Speech-to-Sign (`Text-to-Sign`)**
  - Accepts typed text or live microphone voice input (`SpeechRecognition`) in **English** and **Hindi (Devanagari)**.
  - Drives a 2.5D/3D procedural volumetric hand avatar powered by damped spring physics (`k = 0.30`, `d = 0.52`) across 19 semantic ISL gesture families (`80+` English words and `70+` Hindi words).
  - Automatically falls back to sequential character-by-character fingerspelling (`A–Z`, `0–9`, and Devanagari) for any out-of-vocabulary word.
- **3-Tier Inference Engine**
  - **Tier 1 — Bidirectional LSTM with Attention (`SignLanguageLSTM`):** 2-layer PyTorch BiLSTM (`hidden_size=256`) over 30-frame temporal windows.
  - **Tier 2 — Random Forest Classifier:** 300-tree Scikit-Learn ensemble trained across 251 ISL classes.
  - **Tier 3 — Built-in 3D Geometric Pose Classifier:** Zero-weight geometric posture engine that works immediately out of the box when trained `.pt`/`.pkl` weights are not present.
- **Practice Arena & Analytics (`Insights & Log`)**
  - Interactive calibration arena that computes live 63-dimensional cosine similarity between your webcam posture and target sign vectors.
  - Local transcript archive (`localStorage`) with dynamic activity volume and per-sign precision charts.
- **Accessibility by Design**
  - Light/Dark dual themes, high-contrast mode, adjustable typography scaling (`Normal`, `Large`, `X-Large`), sensory voice feedback, and full keyboard navigation.

---

## Project Structure

```text
SignBridge-Extended/
├── CODE FILES/
│   ├── app.py                 # Flask server & 3-tier inference engine (BiLSTM / RF / Geometric)
│   ├── index.html             # Single-page React 18 + Tailwind CSS frontend application
│   ├── model_classes.json     # 251 ISL class labels (0-9, A-Z, words, and sentences)
│   ├── train.py               # Full dataset MediaPipe landmark extractor & RF trainer
│   ├── train_rf.py            # Fast Random Forest trainer from cached .npz landmarks
│   └── train_lstm.py          # PyTorch Bidirectional LSTM + Attention trainer
├── Document and start files/
│   ├── Start-SignBridge.bat   # One-click Windows batch launcher
│   └── Start-SignBridge.ps1   # PowerShell environment check & server launcher
├── MD FILES/
│   └── ARCHITECTURE.md        # System architecture & data-flow overview
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
└── README.md                  # Project overview & setup guide
```

---

## Getting Started

### Prerequisites
- **Python 3.10+**
- A modern browser with webcam and Web Speech API support (**Google Chrome** or **Microsoft Edge** recommended)

### Option 1: One-Click Windows Startup
Double-click [`Document and start files/Start-SignBridge.bat`](Document%20and%20start%20files/Start-SignBridge.bat) or run from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File ".\Document and start files\Start-SignBridge.ps1"
```

The launcher checks required Python packages, frees port `8000` if occupied, starts the local backend, and opens `http://127.0.0.1:8000` in your default browser.

### Option 2: Manual Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SKYWALKERSID/SignBridge-Extended.git
   cd SignBridge-Extended
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python "CODE FILES/app.py"
   ```

4. **Open in browser**
   Navigate to **http://127.0.0.1:8000**.

---

## Training Custom Models

SignBridge Extended runs out of the box using its built-in **Geometric 3D Pose Classifier**, and automatically upgrades to **Random Forest** or **BiLSTM** as soon as model weights are placed in `CODE FILES/`.

### 1. Extract Landmarks & Train Random Forest (`sign_model.pkl`)
Place your ISL dataset folders under `DATA/` and run:
```bash
pip install opencv-python mediapipe openpyxl
python "CODE FILES/train.py"
```
If you already have cached landmark arrays (`landmarks_clean.npz` or `landmarks_cache.npz` in `CODE FILES/`):
```bash
python "CODE FILES/train_rf.py"
```

### 2. Train the Bidirectional LSTM (`sign_model_lstm.pt`)
```bash
pip install torch
python "CODE FILES/train_lstm.py"
```

---

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Alt + T` | Toggle between Light and Dark themes |
| `Alt + A` | Open / close the Accessibility & Typography panel |
| `Alt + V` | Toggle spoken audio feedback |
| `Escape` | Close active modal dialogs |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
