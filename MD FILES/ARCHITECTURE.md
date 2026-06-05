# SignBridge Architecture

## 1) Purpose

This document defines the architecture for **SignBridge** with two lenses:
- **Current Implementation (as-built in this workspace)**
- **Future Scope (deployment-ready target architecture)**

## 2) Current Implementation (As-Built)

## 2.1 Codebase Snapshot

Current files in workspace:
- `app.py` -> combined frontend + backend web app
- `sign_model.pkl` -> trained classifier artifact
- `collect-data.py` -> data capture script
- `model.py` -> legacy local webcam inference script
- `sign-detection.py` -> scratch/partial detection script
- `PRD.md`, `TRD.md` -> product and technical requirement docs
- `Start-SignBridge.ps1`, `Start-SignBridge.bat` -> launcher scripts

## 2.2 Runtime Architecture (Current)

### Backend
- Framework: **Flask** (single service)
- Model serving: `joblib.load("sign_model.pkl")`
- Inference endpoint:
  - `POST /api/v1/infer/landmarks`
  - Input: 63 landmark float values (1 hand, x/y/z for 21 points)
  - Output: prediction, confidence, smoothed prediction, commit signal
- Text-to-sign endpoint:
  - `POST /api/v1/text-to-sign`
  - Token mapping for English/Hindi phrases to animation ids
- Health endpoint:
  - `GET /api/v1/health`

### Frontend
- Delivered by Flask as inline HTML/CSS/JS from `/`
- Browser webcam access via `getUserMedia`
- Browser-side MediaPipe Hands extracts landmarks
- Frontend sends landmarks to backend inference API in intervals
- Text-to-sign playback uses paper-style subtle UI animation states

### Session/State
- In-memory session smoothing window (`deque`) per `session_id`
- No persistent database in current build

## 2.3 Current Visual/Theming Implementation

- Brand name: **SignBridge**
- Visual System: **Pure Two-Color Dual-Inversion Style**
  - Primary Pale Grey: `#EBEBDF`
  - Accent Rich Blue: `#191265`
- Light Mode: Background is Pale Grey (`#EBEBDF`), structures and text are Rich Blue (`#191265`).
- Dark Mode: Background is Rich Blue (`#191265`), structures and text are Pale Grey (`#EBEBDF`).
- UI style: extremely clean, Swiss-style minimalism, strictly NO neon elements, flashy glows, or third-party accents.
- Theme Switcher: Seamless header-integrated dynamic toggle with smooth color-fade animations.


## 2.4 Current Constraints

- Single-process app (`app.py`) is not horizontally scalable.
- In-memory state is lost on restart.
- No auth/account model.
- No persistent transcript/history storage.
- Limited vocabulary mapping and model classes.
- Font loading currently depends on local/browser fallback unless external font files are integrated.

## 3) Future Scope (Deployment-Ready Target)

## 3.1 Target Logical Architecture

### Client Layer
- **Next.js (React + TypeScript)** web app
- Modules:
  - Live sign-to-text view
  - Text-to-sign animation view
  - Conversation split mode
  - Settings and accessibility

### API Layer
- **FastAPI** service with versioned endpoints under `/api/v1`
- Separate modules for inference, text mapping, health, and model management

### ML Inference Layer
- Model registry and active model loader
- Preprocessing + confidence thresholding + temporal smoothing
- Future optional GPU inference path

### Data Layer
- **PostgreSQL** for persistent product data
- **Redis** for cache/session/ratelimiting/realtime glue
- Object storage (S3-compatible) for model/animation assets

### Edge/Delivery Layer
- Nginx/Caddy reverse proxy
- TLS termination, compression, static caching
- CDN for static assets/animations

## 3.2 Target Deployment Topology

- `frontend` container
- `backend` container
- `postgres` container/service
- `redis` container/service
- optional worker container for async jobs

Orchestrated by Docker Compose (small scale) or Kubernetes (large scale).

## 3.3 Data Model (Future)

Core tables:
- `users` (optional in MVP, required for personalized settings)
- `sign_classes`
- `animation_assets`
- `phrase_mappings`
- `model_versions`
- `inference_events` (anonymized analytics)

## 3.4 API Expansion (Future)

- `POST /api/v1/infer/frame` (image/frame path)
- `POST /api/v1/infer/landmarks` (current compatible)
- `POST /api/v1/text-to-sign`
- `GET /api/v1/animations/{id}`
- `GET /api/v1/models/active`
- `POST /api/v1/models/switch` (admin)

## 3.5 Security/Compliance Scope

- HTTPS-only in production
- Strict CORS and origin policy
- Input schema validation + request size limits
- Rate limiting and abuse protection
- Consent-first webcam workflow
- No raw video retention by default

## 3.6 Observability Scope

- Structured logs
- Error tracking (Sentry)
- Metrics: latency, inference success, confidence distribution, uptime
- Health checks and startup readiness probes

## 4) Migration Path (Current -> Target)

## Phase A: Stabilize Current MVP
- Keep Flask app functional
- Improve model quality and token coverage
- Add basic integration tests

## Phase B: Split Frontend and Backend
- Move UI from inline HTML to Next.js app
- Move APIs from Flask routes into FastAPI service
- Keep `sign_model.pkl` contract stable during migration

## Phase C: Add Persistence + Scale
- Introduce PostgreSQL and Redis
- Add user/session persistence and mapping CRUD
- Add deploy pipeline and environment separation

## Phase D: Production Hardening
- Add auth (if required)
- Add CDN and optimized animation delivery
- Add monitoring, SLOs, automated rollback strategy

## 5) Current vs Future Summary

### Current
- Monolithic Flask app
- Browser MediaPipe + backend prediction
- No DB, no auth, no persistent storage
- Fast local setup and demo-ready

### Future
- Next.js + FastAPI microservice-style split
- PostgreSQL + Redis + object storage
- CI/CD, observability, scalable deployment
- Enterprise-grade reliability and extensibility

## 6) Decision Note: Database Requirement

For **current local MVP usage**, database is not mandatory.

For **deployment-ready production**, database is strongly recommended (and typically required) for:
- mappings/versioning
- operational analytics
- administration workflows
- scalability and maintainability
