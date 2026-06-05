import base64
import json
import os
import time
from collections import Counter, defaultdict, deque

import joblib
import numpy as np
from flask import Flask, jsonify, request, Response

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Optional import guard for production clarity.
try:
    import mediapipe as mp  # noqa: F401
except Exception:
    mp = None

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
        batch_size = x.size(0)
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

MODEL_PATH = "sign_model.pkl"
CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 6
MIN_STABLE_VOTES = 4

app = Flask(__name__)

SCALER_PATH = "scaler.pkl"
scaler = None

# LSTM file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_MODEL_PATH = os.path.join(SCRIPT_DIR, "sign_model_lstm.pt")
LSTM_SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler_lstm.pkl")
LSTM_CLASSES_PATH = os.path.join(SCRIPT_DIR, "model_classes_lstm.json")

lstm_model = None
scaler_lstm = None
CLASSES_LIST = None

try:
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_CLASSES_PATH) and os.path.exists(LSTM_SCALER_PATH):
        with open(LSTM_CLASSES_PATH, "r", encoding="utf-8") as f:
            CLASSES_LIST = json.load(f)
        num_classes = len(CLASSES_LIST)
        
        lstm_model = SignLanguageLSTM(
            input_size=63,
            hidden_size=256,
            num_layers=2,
            num_classes=num_classes,
            bidirectional=True
        )
        checkpoint = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            lstm_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            lstm_model.load_state_dict(checkpoint)
        lstm_model.eval()
        
        scaler_lstm = joblib.load(LSTM_SCALER_PATH)
        ACTIONS = CLASSES_LIST
        print(f"[LSTM] Successfully loaded LSTM model with {num_classes} classes.")
    else:
        print("[WARN] LSTM model files not found. Falling back to Random Forest.")
        lstm_model = None
except Exception as e:
    lstm_model = None
    scaler_lstm = None
    CLASSES_LIST = None
    print(f"[WARN] Failed to load LSTM model ({e}). Falling back to Random Forest.")

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH, mmap_mode='r')
        print(f"[Model] Successfully loaded sign language classifier with memory mapping.")
    else:
        model = None
        print(f"[WARN] {MODEL_PATH} not found. Starting in training/recovery mode.")
        
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"[Scaler] Successfully loaded StandardScaler.")
except Exception as e:
    model = None
    scaler = None
    print(f"[WARN] Failed to load RF model/scaler ({e}).")

if lstm_model is not None:
    ACTIONS = sorted(list(CLASSES_LIST))
elif model is not None and hasattr(model, "classes_"):
    ACTIONS = sorted(list(model.classes_))
else:
    ACTIONS = ["hello", "thanks", "yes", "no", "iloveyou"]

session_windows = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
session_frames = defaultdict(lambda: deque(maxlen=30))

TEXT_TO_ANIM = {
    "en": {
        "hello": "open-palm-wave",
        "thanks": "chin-forward-thank",
        "yes": "fist-nod",
        "no": "index-middle-tap",
        "i": "index-self",
        "love": "cross-arms-heart",
        "you": "point-forward",
    },
    "hi": {
        "??????": "joined-palms-bow",
        "???????": "chin-forward-thank",
        "???": "fist-nod",
        "????": "index-middle-tap",
        "???": "index-self",
        "?????": "cross-arms-heart",
        "??": "point-forward",
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
          <div class="paper-hand" id="paperHand">??</div>
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
    const canvas = document.getElementById('capture');
    const ctx = canvas.getContext('2d');
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
      if (['yes', '???'].includes(t)) return '?';
      if (['no', '????'].includes(t)) return '??';
      if (['thanks', '???????'].includes(t)) return '??';
      if (['iloveyou', 'love', '?????'].includes(t)) return '??';
      return '??';
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
      tokenLine.innerHTML = tokens.map((t) => `<span class=\"token\">${t}</span>`).join('');
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
      paperHand.textContent = '??';
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
        if 0 <= idx < len(ACTIONS):
            return ACTIONS[idx]
    except (TypeError, ValueError):
        pass

    return str(raw_pred)


def api_error(code, message, status=400):
    return jsonify({"error": {"code": code, "message": message, "details": {}}}), status


@app.get("/")
def home():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype="text/html")
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/api/v1/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "version": "1.0.0"})


@app.get("/api/v1/classes")
def get_classes():
    return jsonify({"classes": ACTIONS, "total": len(ACTIONS)})


def normalize_landmarks(landmarks):
    """Normalize a 63-d landmark vector relative to wrist and shape-independent hand size."""
    points = np.array(landmarks).reshape(21, 3)
    
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


@app.post("/api/v1/infer/landmarks")
def infer_landmarks():
    if lstm_model is None and model is None:
        return api_error(
            "MODEL_NOT_READY", 
            "The sign language classification model is currently training/initializing. Please wait...",
            status=503
        )

    payload = request.get_json(silent=True) or {}
    landmarks = payload.get("landmarks")
    session_id = payload.get("session_id", "default")

    if not isinstance(landmarks, list) or len(landmarks) != 63:
        return api_error("INVALID_PAYLOAD", "Expected exactly 63 landmark values")

    try:
        # Apply identical translation and scale normalization
        normalized = normalize_landmarks(landmarks)
        if normalized is None:
            return api_error("INVALID_POSE", "Skeletal scale detection failed (hand invalid or too far)")
            
        if lstm_model is not None:
            # 1. Scale frame features using scaler_lstm
            features = np.array(normalized, dtype=np.float32).reshape(1, -1)
            if scaler_lstm is not None:
                features = scaler_lstm.transform(features)
            # Flatten back to (63,)
            scaled_frame = features[0]
            
            # 2. Append to sliding window
            session_frames[session_id].append(scaled_frame)
            
            # 3. Create padded input sequence (MAX_SEQ_LEN = 30)
            frames_list = list(session_frames[session_id])
            actual_len = len(frames_list)
            
            padded = np.zeros((30, 63), dtype=np.float32)
            padded[:actual_len] = frames_list
            
            # 4. Prepare tensors
            x_tensor = torch.FloatTensor(padded).unsqueeze(0)  # (1, 30, 63)
            len_tensor = torch.LongTensor([actual_len])
            
            # 5. Model forward pass
            with torch.no_grad():
                logits = lstm_model(x_tensor, len_tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                pred_idx = np.argmax(probs)
                confidence = float(probs[pred_idx])
                pred_label = CLASSES_LIST[pred_idx]
        else:
            # Random Forest Fallback
            features = np.array(normalized, dtype=np.float32).reshape(1, -1)
            if scaler is not None:
                features = scaler.transform(features)
            pred_raw = model.predict(features)[0]
            pred_label = normalize_prediction(pred_raw)

            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(features)))
            else:
                confidence = 0.5

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
            "top_k": [{"label": pred_label, "confidence": confidence}],
            "smoothed_prediction": smoothed,
            "commit": bool(commit),
            "ts": int(time.time() * 1000),
        }
    )


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
    tokens = normalized.lower().split() if language == "en" else normalized.split()

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
