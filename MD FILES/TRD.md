# Technical Requirements Document (TRD)

## 1) Technical Scope and Assumptions

This TRD defines a production-oriented architecture for an ISL translator web platform based on the existing Python scratch assets in this workspace.

### Existing Artifacts (Current State)
- `collect-data.py`: webcam data collection for fixed class list.
- `model.py` + `sign_model.pkl`: trained classifier artifact (landmark-based).
- `sign-detection.py`: local OpenCV + MediaPipe real-time inference demo.

### Key Assumptions
- Color palette confirmed: pale grey `#EBEBDF` and rich blue `#191265`.
- Phase 1 sign recognition is classifier-based on static/short gestures, not full continuous sign-language grammar parsing.
- Text-to-sign animation uses token/phrase asset mapping rather than generative avatar synthesis in MVP.

## 2) High-Level Architecture

### Frontend (Web Client)
- Framework: **Next.js (React + TypeScript)**
- Styling: CSS variables + Tailwind or CSS Modules
- Realtime transport options:
  - Default: HTTP polling for MVP simplicity
  - Preferred: WebSocket for low-latency inference streaming
- Camera access: `navigator.mediaDevices.getUserMedia`
- Canvas/WebGL layer for overlay rendering and paper-style animation stage

### Backend (Python API)
- Framework: **FastAPI**
- Inference runtime:
  - MediaPipe hands landmark extraction (if server-side frame pipeline used)
  - `scikit-learn`/joblib model load for `sign_model.pkl`
- Async server: Uvicorn/Gunicorn
- Optional queue for heavy jobs (future): Redis + RQ/Celery

### Data/Storage
- PostgreSQL for:
  - users (optional for MVP)
  - phrase dictionaries
  - animation metadata
  - model versions and telemetry summaries
- Object storage (S3-compatible) for animation assets and optional datasets

### DevOps/Infra
- Dockerized services:
  - `frontend`
  - `backend`
  - `postgres`
- Reverse proxy: Nginx/Caddy
- Deploy targets: Render/Railway/Fly/VM/Kubernetes
- HTTPS via managed certs

## 3) Detailed Component Design

## 3.1 Frontend Modules

- `app/(routes)/translator`: live sign-to-text page.
- `app/(routes)/text-to-sign`: typed text to animation page.
- `components/camera`: webcam controls and permission UX.
- `components/prediction`: confidence chips + rolling transcript.
- `components/animation-stage`: paper-style sign animation renderer.
- `lib/api-client`: typed API client (Zod/OpenAPI-generated types).
- `lib/i18n`: Hindi/English localization resources.
- `lib/theme`: CSS variables for palette + typography.

## 3.2 Backend Modules

- `api/routes/inference.py`
  - frame/landmark coordinate capture and processing
  - prediction and smoothing state handling
- `api/routes/text_to_sign.py`
  - text normalization
  - tokenization
  - phrase->animation mapping
- `api/routes/health.py`
- `core/model_registry.py`
  - load, switch, validate model artifacts
- `core/inference_engine.py`
  - preprocess, predict, thresholding
- `core/temporal_smoother.py`
  - rolling window vote/confidence smoothing
- `core/animation_mapper.py`
  - text token to animation keyframe sequence
- `db/models/*.py`
- `schemas/*.py` (Pydantic request/response contracts)

## 4) API Specification (MVP)

## 4.1 Health

### `GET /api/v1/health`
Response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

## 4.2 Frame Inference

### `POST /api/v1/infer/frame`
Request (option A - image):
```json
{
  "image_base64": "...",
  "session_id": "uuid",
  "timestamp_ms": 1716200000000
}
```

Request (option B - landmarks from client-side MediaPipe):
```json
{
  "landmarks": [0.12, -0.03, 0.04, "..."],
  "session_id": "uuid",
  "timestamp_ms": 1716200000000
}
```

Response:
```json
{
  "prediction": "hello",
  "confidence": 0.93,
  "top_k": [
    {"label": "hello", "confidence": 0.93},
    {"label": "thanks", "confidence": 0.04},
    {"label": "yes", "confidence": 0.02}
  ],
  "smoothed_prediction": "hello",
  "commit": true
}
```

## 4.3 Text to Sign Sequence

### `POST /api/v1/text-to-sign`
Request:
```json
{
  "text": "?????? ?? ???? ???",
  "language": "hi"
}
```

Response:
```json
{
  "normalized_text": "?????? ?? ???? ???",
  "tokens": ["??????", "??", "????", "???"],
  "sequence": [
    {"token": "??????", "animation_id": "hi_namaste_001", "duration_ms": 1200},
    {"token": "??", "animation_id": "hi_aap_001", "duration_ms": 900}
  ],
  "fallback_tokens": ["????", "???"]
}
```

## 4.4 Animation Metadata

### `GET /api/v1/animations/{animation_id}`
Response:
```json
{
  "animation_id": "hi_namaste_001",
  "asset_url": "https://cdn.example.com/animations/hi_namaste_001.json",
  "style": "paper-subtle",
  "fps": 24
}
```

## 5) Data Model (Relational)

### `sign_classes`
- `id` (PK)
- `label`
- `language` (`en`/`hi`/`universal`)
- `is_active`

### `animation_assets`
- `id` (PK)
- `animation_key` (unique)
- `language`
- `asset_url`
- `style` (must include `paper-subtle`)
- `duration_ms`

### `phrase_mappings`
- `id` (PK)
- `phrase`
- `language`
- `token_sequence` (JSONB)
- `animation_sequence` (JSONB)

### `model_versions`
- `id` (PK)
- `name`
- `artifact_path`
- `metrics` (JSONB)
- `is_active`
- `created_at`

## 6) ML and Inference Pipeline

## 6.1 Offline Training
- Data capture from controlled sessions (expand beyond current five labels).
- Landmark extraction normalization.
- Train baseline classifier (RandomForest/SVM/GradientBoosting baseline).
- Evaluate confusion matrix + per-class precision/recall.
- Store artifact with semantic version.

## 6.2 Online Inference
1. Receive frame or landmarks.
2. If frame: detect hand landmarks with MediaPipe.
3. Flatten to feature vector.
4. Predict class probabilities from active model.
5. Apply threshold + temporal smoothing window.
6. Return top-k and commit signal.

## 6.3 Confidence and Commit Logic
- `confidence_threshold`: e.g. 0.75
- `min_stable_frames`: e.g. 4 out of rolling 6
- Emit `commit=true` only when both confidence and stability pass.

## 7) Frontend Visual/Theming Spec

### pure Two-Color System
- **Pale Grey:** `#EBEBDF`
- **Rich Blue:** `#191265`

### CSS Theme Variables
```css
/* Light Theme (Default) */
.theme-light {
  --color-bg: #EBEBDF;
  --color-fg: #191265;
  --color-surface: #F5F5EE;
  --color-border: #191265;
  --color-card-bg: #FFFFFF;
}

/* Dark Theme (Pure Inversion) */
.theme-dark {
  --color-bg: #191265;
  --color-fg: #EBEBDF;
  --color-surface: #1E1773;
  --color-border: #EBEBDF;
  --color-card-bg: #160F5C;
}
```

### Typography
- **Heading/Brand font:** `"Peace Sans"`, `"Space Grotesk"`, sans-serif
- **Body/UI font:** `"Open Sauce One"`, `"Plus Jakarta Sans"`, `"Inter"`, sans-serif

### Motion Tokens
- Theme toggle transition: `background-color 400ms ease, color 400ms ease`
- Layout reveal: `duration-300 easing-out`
- Easing: `cubic-bezier(0.16, 1, 0.3, 1)`


## 8) Security, Privacy, and Compliance

- Camera permission prompt and clear consent UI.
- Privacy notice explaining frame processing policy.
- Disable raw frame persistence by default.
- JWT-based auth only if user accounts are enabled.
- Basic rate limiting on inference endpoints.
- Input validation with strict schema checks.

## 9) Cross-Platform and Browser Strategy

- Responsive design with touch-safe controls.
- Fallback UI when camera APIs are unavailable.
- Safari/iOS testing for autoplay/camera restrictions.
- Progressive enhancement: core features work without WebGL extras.

## 10) Testing Requirements

### Backend
- Unit tests for preprocessing, smoothing, threshold logic.
- Contract tests for each API endpoint.
- Load test baseline for inference endpoint.

### Frontend
- Component tests for translator and animation controls.
- E2E tests (Playwright) for permission flow and full translation loop.
- Accessibility tests (keyboard + ARIA smoke tests).

### Model
- Regression test set with known landmark vectors.
- Drift checks across model version upgrades.

## 11) Deployment and Environments

### Environments
- `dev`: local Docker Compose
- `staging`: production-like test stack
- `prod`: autoscaled API and CDN-served static assets

### CI/CD
- Lint + tests + type checks.
- Build Docker images.
- Deploy backend then frontend.
- Run post-deploy smoke tests:
  - health endpoint
  - one inference call
  - one text-to-sign call

## 12) Error Handling Contract

Error response format:
```json
{
  "error": {
    "code": "INFERENCE_NO_HAND",
    "message": "No detectable hand landmarks found",
    "details": {}
  }
}
```

Common codes:
- `INVALID_PAYLOAD`
- `CAMERA_PERMISSION_REQUIRED`
- `INFERENCE_NO_HAND`
- `MODEL_NOT_LOADED`
- `ANIMATION_TOKEN_UNMAPPED`

## 13) Implementation Roadmap (Execution-Ready)

### Sprint 1
- Repository scaffolding: Next.js + FastAPI + Docker Compose.
- Health endpoint, model loading, basic inference endpoint.
- Base theme with required palette and fonts.

### Sprint 2
- Webcam UI, live prediction panel, smoothing integration.
- Hindi/English i18n base and transcript controls.

### Sprint 3
- Text-to-sign tokenization and animation sequence service.
- Paper-style subtle animation renderer + controls.

### Sprint 4
- Accessibility hardening, cross-browser testing, performance tuning.
- Staging deploy and acceptance validation.

## 14) MVP Exit Criteria

- Stable cross-platform website deployment.
- Sign-to-text live webcam loop operational.
- Text-to-sign animation loop operational.
- API contracts documented and validated.
- UI meets minimalist pastel constraints and avoids neon aesthetics.

