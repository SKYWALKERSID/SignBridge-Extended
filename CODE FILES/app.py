import json
import os
import re
import time
from collections import Counter, defaultdict, deque

import joblib
import numpy as np
from flask import Flask, Response, jsonify, request

# Optional PyTorch import so the server still runs in Random Forest or Geometric mode
try:
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    pack_padded_sequence = None
    pad_packed_sequence = None
    TORCH_AVAILABLE = False

# Optional MediaPipe import guard for production clarity
try:
    import mediapipe as mp  # noqa: F401
except Exception:
    mp = None


if TORCH_AVAILABLE:
    class SignLanguageLSTM(nn.Module):
        """
        Bidirectional LSTM with attention pooling for sign language classification.
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
else:
    SignLanguageLSTM = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(SCRIPT_DIR, "index.html")

# Random Forest file paths (anchored to SCRIPT_DIR)
MODEL_PATH = os.path.join(SCRIPT_DIR, "sign_model.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler.pkl")
RF_CLASSES_PATH = os.path.join(SCRIPT_DIR, "model_classes.json")

# LSTM file paths (anchored to SCRIPT_DIR)
LSTM_MODEL_PATH = os.path.join(SCRIPT_DIR, "sign_model_lstm.pt")
LSTM_SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler_lstm.pkl")
LSTM_CLASSES_PATH = os.path.join(SCRIPT_DIR, "model_classes_lstm.json")

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 6
MIN_STABLE_VOTES = 4

app = Flask(__name__)

lstm_model = None
scaler_lstm = None
CLASSES_LIST = None

model = None
scaler = None
rf_classes_list = None

# Load fallback class list from model_classes.json if available
if os.path.exists(RF_CLASSES_PATH):
    try:
        with open(RF_CLASSES_PATH, "r", encoding="utf-8") as f:
            loaded_classes = json.load(f)
            if isinstance(loaded_classes, list) and loaded_classes:
                rf_classes_list = [str(c) for c in loaded_classes]
    except Exception as e:
        print(f"[WARN] Could not parse {RF_CLASSES_PATH}: {e}")

# 1. Try loading LSTM model if PyTorch and artifacts are available
if TORCH_AVAILABLE:
    try:
        if (
            os.path.exists(LSTM_MODEL_PATH)
            and os.path.exists(LSTM_CLASSES_PATH)
            and os.path.exists(LSTM_SCALER_PATH)
        ):
            with open(LSTM_CLASSES_PATH, "r", encoding="utf-8") as f:
                CLASSES_LIST = json.load(f)
            num_classes = len(CLASSES_LIST)

            lstm_model = SignLanguageLSTM(
                input_size=63,
                hidden_size=256,
                num_layers=2,
                num_classes=num_classes,
                bidirectional=True,
            )
            checkpoint = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                lstm_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                lstm_model.load_state_dict(checkpoint)
            lstm_model.eval()

            scaler_lstm = joblib.load(LSTM_SCALER_PATH)
            print(f"[LSTM] Successfully loaded LSTM model with {num_classes} classes.")
        else:
            print("[INFO] LSTM model files not found. Checking Random Forest model...")
    except Exception as e:
        lstm_model = None
        scaler_lstm = None
        CLASSES_LIST = None
        print(f"[WARN] Failed to load LSTM model ({e}). Falling back to Random Forest.")
else:
    print("[INFO] PyTorch is not installed. Skipping LSTM loader.")

# 2. Try loading Random Forest model
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH, mmap_mode="r")
        print("[Model] Successfully loaded Random Forest classifier with memory mapping.")
    else:
        model = None
        print(f"[INFO] {os.path.basename(MODEL_PATH)} not found. Using built-in geometric hand-pose classifier.")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("[Scaler] Successfully loaded StandardScaler.")
except Exception as e:
    model = None
    scaler = None
    print(f"[WARN] Failed to load RF model/scaler ({e}).")

# Determine active vocabulary list
if lstm_model is not None and CLASSES_LIST:
    ACTIONS = sorted(list(CLASSES_LIST))
elif model is not None and hasattr(model, "classes_"):
    raw_model_classes = list(model.classes_)
    if rf_classes_list and len(rf_classes_list) == len(raw_model_classes):
        ACTIONS = rf_classes_list
    else:
        ACTIONS = [str(c) for c in sorted(raw_model_classes)]
elif rf_classes_list:
    ACTIONS = rf_classes_list
else:
    ACTIONS = ["hello", "thanks", "yes", "no", "iloveyou"]

session_windows = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
session_frames = defaultdict(lambda: deque(maxlen=30))

TEXT_TO_ANIM = {
    "en": {
        "hello": "open-palm-wave",
        "hi": "open-palm-wave",
        "namaste": "joined-palms-bow",
        "thanks": "chin-forward-thank",
        "thank": "chin-forward-thank",
        "yes": "fist-nod",
        "no": "index-middle-tap",
        "i": "index-self",
        "me": "index-self",
        "my": "index-self",
        "love": "cross-arms-heart",
        "iloveyou": "cross-arms-heart",
        "you": "point-forward",
        "please": "open-palm-wave",
        "sorry": "fist-nod",
        "help": "chin-forward-thank",
        "good": "fist-nod",
        "stop": "open-palm-wave",
    },
    "hi": {
        "नमस्ते": "joined-palms-bow",
        "नमस्कार": "joined-palms-bow",
        "धन्यवाद": "chin-forward-thank",
        "शुक्रिया": "chin-forward-thank",
        "हाँ": "fist-nod",
        "हां": "fist-nod",
        "नहीं": "index-middle-tap",
        "ना": "index-middle-tap",
        "मैं": "index-self",
        "मेरा": "index-self",
        "प्यार": "cross-arms-heart",
        "प्रेम": "cross-arms-heart",
        "तुम": "point-forward",
        "आप": "point-forward",
        "कृपया": "open-palm-wave",
        "माफ़": "fist-nod",
        "मदद": "chin-forward-thank",
        "अच्छा": "fist-nod",
        "रुको": "open-palm-wave",
    },
}

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SignBridge</title>
  <style>
    :root {
      --bg: #EBEBDF;
      --primary: #191265;
      --surface: #f7f7ef;
      --border: #d6d6ca;
      --text: #1c1c2c;
      --muted: #5f5f76;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 20% 0%, #f3f3ea, var(--bg) 35%);
      color: var(--text);
      font-family: "Open Sauce One", "Segoe UI", system-ui, sans-serif;
      min-height: 100vh;
    }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .title {
      font-family: "Peace Sans", "Open Sauce One", sans-serif;
      letter-spacing: 0.5px;
      color: var(--primary);
      margin: 0;
    }
    .sub { color: var(--muted); margin: 8px 0 20px; }
    .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }
    .card {
      background: linear-gradient(180deg, #ffffffb8, var(--surface));
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 10px 26px rgba(25, 18, 101, 0.06);
      animation: paperIn .45s ease;
    }
    @keyframes paperIn {
      from { opacity: 0; transform: translateY(8px) scale(0.995); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
    button, select, input, textarea {
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
    }
    button {
      cursor: pointer;
      transition: transform .2s ease, background .2s ease;
    }
    button:hover { transform: translateY(-1px); background: #f4f4fc; }
    video { width: 100%; border-radius: 12px; border: 1px solid var(--border); transform: scaleX(-1); }
    .pred {
      margin-top: 10px;
      padding: 10px;
      background: #fff;
      border: 1px dashed var(--border);
      border-radius: 10px;
    }
    .pill { display: inline-block; background: #ebe9fb; color: var(--primary); padding: 2px 8px; border-radius: 999px; font-size: 12px; }
    .anim-stage {
      min-height: 180px;
      border: 1px dashed var(--border);
      border-radius: 12px;
      background: repeating-linear-gradient(0deg, #fafaf3, #fafaf3 18px, #f4f4eb 18px, #f4f4eb 36px);
      position: relative;
      overflow: hidden;
      padding: 12px;
    }
    .paper-hand {
      position: absolute;
      left: 14px;
      top: 54px;
      width: 88px;
      height: 88px;
      border-radius: 16px;
      background: #fff;
      border: 1px solid var(--border);
      display: grid;
      place-items: center;
      font-size: 42px;
      animation: floaty 2.8s ease-in-out infinite;
    }
    @keyframes floaty {
      0%,100% { transform: translateY(0) rotate(-1deg); }
      50% { transform: translateY(-7px) rotate(1deg); }
    }
    .anim-line { margin-left: 120px; padding: 10px; color: var(--primary); }
    .token {
      display: inline-block;
      margin: 4px 6px 0 0;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #fff;
      opacity: .75;
      transform: translateY(0);
      transition: all .24s ease;
    }
    .token.active {
      opacity: 1;
      background: #ecebfc;
      border-color: #c8c4f5;
      transform: translateY(-2px);
    }
    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">SignBridge</h1>
    <p class="sub">ISL sign-to-text and text-to-sign translator (Hindi + English)</p>

    <div class="grid">
      <section class="card">
        <h3>Live Sign to Text</h3>
        <div class="controls">
          <button id="startCamera">Start Camera</button>
          <button id="stopCamera">Stop Camera</button>
          <span class="pill" id="connState">idle</span>
        </div>
        <video id="video" autoplay muted playsinline></video>
        <canvas id="capture" width="320" height="240" style="display:none"></canvas>
        <div class="pred">
          <div><strong>Prediction:</strong> <span id="predText">-</span></div>
          <div><strong>Confidence:</strong> <span id="predConf">-</span></div>
          <div><strong>Transcript:</strong> <span id="transcript">-</span></div>
        </div>
      </section>

      <section class="card">
        <h3>Text to Sign Animation</h3>
        <div class="controls">
          <select id="lang">
            <option value="en">English</option>
            <option value="hi">Hindi</option>
          </select>
          <button id="runAnim">Animate</button>
          <button id="clearAnim">Clear</button>
        </div>
        <textarea id="textInput" rows="3" style="width:100%" placeholder="Type text here..."></textarea>
        <div class="anim-stage" id="animStage">
          <div class="paper-hand" id="paperHand">👋</div>
          <div class="anim-line" id="animLine">Waiting for input...</div>
          <div class="anim-line" id="tokenLine"></div>
        </div>
      </section>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js"></script>
  <script>
    const video = document.getElementById('video');
    const predText = document.getElementById('predText');
    const predConf = document.getElementById('predConf');
    const transcript = document.getElementById('transcript');
    const connState = document.getElementById('connState');

    const animLine = document.getElementById('animLine');
    const tokenLine = document.getElementById('tokenLine');
    const paperHand = document.getElementById('paperHand');

    let mediaStream = null;
    let loopTimer = null;
    let transcriptWords = [];

    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5
    });

    let latestLandmarks = null;
    hands.onResults((results) => {
      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const lm = results.multiHandLandmarks[0];
        latestLandmarks = lm.flatMap((pt) => [pt.x, pt.y, pt.z]);
      } else {
        latestLandmarks = null;
      }
    });

    async function tickInference() {
      if (!mediaStream) return;
      try {
        await hands.send({ image: video });
        if (!latestLandmarks || latestLandmarks.length !== 63) {
          predText.textContent = 'No hand detected';
          predConf.textContent = '-';
          return;
        }

        const response = await fetch('/api/v1/infer/landmarks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ landmarks: latestLandmarks, session_id: 'web-session-1' })
        });

        const data = await response.json();
        if (!response.ok) {
          predText.textContent = data?.error?.message || 'Inference failed';
          return;
        }

        predText.textContent = data.smoothed_prediction || data.prediction || '-';
        predConf.textContent = (data.confidence * 100).toFixed(1) + '%';

        if (data.commit && data.smoothed_prediction) {
          const last = transcriptWords[transcriptWords.length - 1];
          if (last !== data.smoothed_prediction) {
            transcriptWords.push(data.smoothed_prediction);
            transcript.textContent = transcriptWords.join(' ');
          }
        }
      } catch (e) {
        predText.textContent = 'Inference error';
      }
    }

    document.getElementById('startCamera').addEventListener('click', async () => {
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        video.srcObject = mediaStream;
        await video.play();
        connState.textContent = 'running';
        clearInterval(loopTimer);
        loopTimer = setInterval(tickInference, 260);
      } catch (err) {
        connState.textContent = 'camera denied';
      }
    });

    document.getElementById('stopCamera').addEventListener('click', () => {
      if (mediaStream) {
        mediaStream.getTracks().forEach((t) => t.stop());
      }
      mediaStream = null;
      clearInterval(loopTimer);
      connState.textContent = 'stopped';
    });

    function handForToken(token) {
      const t = token.toLowerCase();
      if (['yes', 'हाँ', 'हां'].includes(t)) return '✊';
      if (['no', 'नहीं', 'ना'].includes(t)) return '✌️';
      if (['thanks', 'thank', 'धन्यवाद', 'शुक्रिया'].includes(t)) return '🙏';
      if (['iloveyou', 'love', 'प्यार', 'प्रेम'].includes(t)) return '🤟';
      return '👋';
    }

    async function animateText() {
      const text = document.getElementById('textInput').value.trim();
      const language = document.getElementById('lang').value;
      if (!text) return;

      const response = await fetch('/api/v1/text-to-sign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language })
      });
      const data = await response.json();
      if (!response.ok) {
        animLine.textContent = data?.error?.message || 'Mapping failed';
        return;
      }

      const tokens = data.tokens || [];
      tokenLine.innerHTML = tokens.map((t) => `<span class="token">${t}</span>`).join('');
      animLine.textContent = 'Playing paper-style sequence...';

      const tokenEls = [...tokenLine.querySelectorAll('.token')];
      for (let i = 0; i < tokens.length; i += 1) {
        tokenEls.forEach((el) => el.classList.remove('active'));
        tokenEls[i].classList.add('active');
        paperHand.textContent = handForToken(tokens[i]);
        await new Promise((r) => setTimeout(r, 700));
      }
      tokenEls.forEach((el) => el.classList.remove('active'));
      animLine.textContent = data.fallback_tokens.length
        ? `Done. Missing tokens: ${data.fallback_tokens.join(', ')}`
        : 'Done. All tokens mapped.';
    }

    document.getElementById('runAnim').addEventListener('click', animateText);
    document.getElementById('clearAnim').addEventListener('click', () => {
      document.getElementById('textInput').value = '';
      animLine.textContent = 'Waiting for input...';
      tokenLine.textContent = '';
      paperHand.textContent = '👋';
    });
  </script>
</body>
</html>
"""


def normalize_prediction(raw_pred):
    if isinstance(raw_pred, str):
        return raw_pred

    try:
        idx = int(raw_pred)
        if rf_classes_list and 0 <= idx < len(rf_classes_list):
            return rf_classes_list[idx]
        if 0 <= idx < len(ACTIONS):
            return ACTIONS[idx]
    except (TypeError, ValueError):
        pass

    return str(raw_pred)


def api_error(code, message, status=400):
    return jsonify({"error": {"code": code, "message": message, "details": {}}}), status


@app.get("/")
def home():
    if os.path.exists(INDEX_HTML_PATH):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype="text/html")
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/api/v1/health")
def health():
    engine = (
        "lstm"
        if lstm_model is not None
        else ("random_forest" if model is not None else "geometric_fallback")
    )
    return jsonify(
        {
            "status": "ok",
            "model_loaded": bool(lstm_model is not None or model is not None),
            "engine": engine,
            "classes_count": len(ACTIONS),
            "version": "1.1.0",
        }
    )


@app.get("/api/v1/classes")
def get_classes():
    return jsonify({"classes": ACTIONS, "total": len(ACTIONS)})


def normalize_landmarks(landmarks):
    """Normalize a 63-d landmark vector relative to wrist and shape-independent hand size."""
    points = np.array(landmarks, dtype=np.float32).reshape(21, 3)

    # Step 1: Translate to wrist origin (translation invariance)
    wrist = points[0].copy()
    points = points - wrist

    # Step 2: Scale normalize using the rigid wrist-to-middle-MCP joint distance (points[9])
    hand_size = np.linalg.norm(points[9])
    if hand_size < 1e-6:
        return None

    points = points / hand_size

    # Step 3: Clip extreme outliers to prevent wild fluctuations
    points = np.clip(points, -3.0, 3.0)
    return points.flatten().tolist()


def classify_landmarks_geometric(normalized_flat):
    """
    Built-in geometric 3D hand-pose classifier used when external .pt/.pkl weights
    are not present on disk. Evaluates finger extension ratios, inter-tip distances,
    and palm orientation on wrist-normalized 21-joint MediaPipe coordinates.
    """
    pts = np.array(normalized_flat, dtype=np.float32).reshape(21, 3)

    def dist(a, b):
        return float(np.linalg.norm(pts[a] - pts[b]))

    # Finger extension checks (tip further from wrist & MCP than PIP)
    thumb_ext = dist(4, 17) > dist(3, 17) * 1.12 and dist(4, 5) > 0.55
    index_ext = dist(8, 0) > dist(6, 0) * 1.15 and dist(8, 5) > dist(6, 5) * 1.20
    middle_ext = dist(12, 0) > dist(10, 0) * 1.15 and dist(12, 9) > dist(10, 9) * 1.20
    ring_ext = dist(16, 0) > dist(14, 0) * 1.12 and dist(16, 13) > dist(14, 13) * 1.18
    pinky_ext = dist(20, 0) > dist(18, 0) * 1.12 and dist(20, 17) > dist(18, 17) * 1.18

    thumb_index_pinch = dist(4, 8) < 0.42
    spread_index_middle = dist(8, 12)
    palm_spread = dist(8, 20)

    candidates = []

    # 1. I Love You (Thumb + Index + Pinky extended, Middle + Ring curled)
    if thumb_ext and index_ext and not middle_ext and not ring_ext and pinky_ext:
        candidates = [("iloveyou", 0.94), ("I", 0.04), ("Y", 0.02)]
    # 2. All 5 fingers extended (Hello / Thanks / Stop / 5)
    elif thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
        if palm_spread > 1.05:
            candidates = [("hello", 0.91), ("5", 0.06), ("STOP", 0.03)]
        else:
            candidates = [("thanks", 0.89), ("hello", 0.08), ("B", 0.03)]
    # 3. Four fingers extended, thumb tucked
    elif not thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
        if palm_spread < 0.85:
            candidates = [("B", 0.88), ("thanks", 0.08), ("4", 0.04)]
        else:
            candidates = [("4", 0.89), ("hello", 0.07), ("B", 0.04)]
    # 4. Thumb + Index pinched with Middle, Ring, Pinky extended (F / Fine)
    elif thumb_index_pinch and middle_ext and ring_ext and pinky_ext:
        candidates = [("FINE", 0.90), ("F", 0.07), ("GOOD", 0.03)]
    # 5. Index + Middle extended, Ring + Pinky curled (No / V / 2)
    elif index_ext and middle_ext and not ring_ext and not pinky_ext:
        if spread_index_middle < 0.42 or dist(4, 12) < 0.55:
            candidates = [("no", 0.90), ("U", 0.06), ("2", 0.04)]
        else:
            candidates = [("V", 0.88), ("2", 0.08), ("no", 0.04)]
    # 6. Index + Middle + Ring extended (3 / W)
    elif index_ext and middle_ext and ring_ext and not pinky_ext:
        candidates = [("W", 0.87), ("3", 0.09), ("hello", 0.04)]
    # 7. Single Index extended (You / 1 / D)
    elif index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if abs(pts[8][0] - pts[5][0]) > abs(pts[8][1] - pts[5][1]) * 0.75:
            candidates = [("you", 0.89), ("1", 0.07), ("D", 0.04)]
        else:
            candidates = [("1", 0.88), ("D", 0.07), ("you", 0.05)]
    # 8. Thumb + Pinky extended (Y / Phone)
    elif thumb_ext and not index_ext and not middle_ext and not ring_ext and pinky_ext:
        candidates = [("Y", 0.91), ("PHONE", 0.06), ("iloveyou", 0.03)]
    # 9. Pinky only extended (I)
    elif not thumb_ext and not index_ext and not middle_ext and not ring_ext and pinky_ext:
        candidates = [("I", 0.90), ("J", 0.06), ("1", 0.04)]
    # 10. Thumb only extended (Good / A)
    elif thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if pts[4][1] < pts[2][1] - 0.15:
            candidates = [("GOOD", 0.90), ("A", 0.06), ("yes", 0.04)]
        else:
            candidates = [("A", 0.87), ("yes", 0.09), ("S", 0.04)]
    # 11. Closed Fist (Yes / S / 0)
    elif not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if thumb_index_pinch:
            candidates = [("O", 0.87), ("0", 0.08), ("yes", 0.05)]
        else:
            candidates = [("yes", 0.91), ("S", 0.05), ("A", 0.04)]
    else:
        candidates = [("hello", 0.76), ("thanks", 0.14), ("yes", 0.10)]

    pred_label, confidence = candidates[0]
    top_k = [{"label": lbl, "confidence": conf} for lbl, conf in candidates]
    return pred_label, confidence, top_k


@app.post("/api/v1/infer/landmarks")
def infer_landmarks():
    payload = request.get_json(silent=True) or {}
    landmarks = payload.get("landmarks")
    session_id = payload.get("session_id", "default")

    if not isinstance(landmarks, list) or len(landmarks) != 63:
        return api_error("INVALID_PAYLOAD", "Expected exactly 63 landmark values")

    try:
        normalized = normalize_landmarks(landmarks)
        if normalized is None:
            return api_error("INVALID_POSE", "Skeletal scale detection failed (hand invalid or too far)")

        if lstm_model is not None and TORCH_AVAILABLE:
            features = np.array(normalized, dtype=np.float32).reshape(1, -1)
            if scaler_lstm is not None:
                features = scaler_lstm.transform(features)
            scaled_frame = features[0]

            session_frames[session_id].append(scaled_frame)
            frames_list = list(session_frames[session_id])
            actual_len = len(frames_list)

            padded = np.zeros((30, 63), dtype=np.float32)
            padded[:actual_len] = frames_list

            x_tensor = torch.FloatTensor(padded).unsqueeze(0)
            len_tensor = torch.LongTensor([actual_len])

            with torch.no_grad():
                logits = lstm_model(x_tensor, len_tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                top_indices = np.argsort(probs)[::-1][:3]
                pred_idx = int(top_indices[0])
                confidence = float(probs[pred_idx])
                pred_label = CLASSES_LIST[pred_idx]
                top_k = [
                    {"label": CLASSES_LIST[int(i)], "confidence": float(probs[int(i)])}
                    for i in top_indices
                ]
        elif model is not None:
            features = np.array(normalized, dtype=np.float32).reshape(1, -1)
            if scaler is not None:
                features = scaler.transform(features)
            pred_raw = model.predict(features)[0]
            pred_label = normalize_prediction(pred_raw)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                top_indices = np.argsort(probs)[::-1][:3]
                confidence = float(probs[top_indices[0]])
                raw_classes = list(model.classes_)
                top_k = [
                    {
                        "label": normalize_prediction(raw_classes[int(i)]),
                        "confidence": float(probs[int(i)]),
                    }
                    for i in top_indices
                ]
            else:
                confidence = 0.80
                top_k = [{"label": pred_label, "confidence": confidence}]
        else:
            pred_label, confidence, top_k = classify_landmarks_geometric(normalized)

    except Exception as exc:
        return api_error("INFERENCE_FAILED", f"Inference failed: {exc}", status=500)

    session_windows[session_id].append(pred_label)
    counts = Counter(session_windows[session_id])
    smoothed, stable_count = counts.most_common(1)[0]
    commit = confidence >= CONFIDENCE_THRESHOLD and stable_count >= MIN_STABLE_VOTES

    return jsonify(
        {
            "prediction": pred_label,
            "confidence": confidence,
            "top_k": top_k,
            "smoothed_prediction": smoothed,
            "commit": bool(commit),
            "ts": int(time.time() * 1000),
        }
    )


def clean_token(raw_token, language):
    """Strip surrounding punctuation while keeping letters, numbers, and Devanagari."""
    if language == "en":
        return re.sub(r"^[^\w]+|[^\w]+$", "", raw_token.lower())
    return re.sub(r'^[.,!?;:"\'()\[\]{}।-]+|[.,!?;:"\'()\[\]{}।-]+$', "", raw_token)


@app.post("/api/v1/text-to-sign")
def text_to_sign():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    language = payload.get("language", "en")

    if not isinstance(text, str) or not text.strip():
        return api_error("INVALID_PAYLOAD", "Text is required")

    if language not in ("en", "hi"):
        return api_error("INVALID_PAYLOAD", "language must be 'en' or 'hi'")

    normalized = " ".join(text.strip().split())
    raw_tokens = normalized.split()
    tokens = [t for t in (clean_token(rt, language) for rt in raw_tokens) if t]

    mapping = TEXT_TO_ANIM[language]
    sequence = []
    fallback_tokens = []

    for token in tokens:
        animation_id = mapping.get(token)
        if animation_id:
            sequence.append({"token": token, "animation_id": animation_id, "duration_ms": 700})
        else:
            fallback_tokens.append(token)

    return jsonify(
        {
            "normalized_text": normalized,
            "tokens": tokens,
            "sequence": sequence,
            "fallback_tokens": fallback_tokens,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
