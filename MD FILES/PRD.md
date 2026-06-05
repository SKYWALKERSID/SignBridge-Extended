# Product Requirements Document (PRD)

## 1) Product Overview

**Product Name:** SignBridge  
**Type:** Cross-platform web application  
**Primary Purpose:** Real-time communication bridge for Indian Sign Language (ISL) users and non-sign users through:
- **Sign-to-Text** (webcam input -> predicted sign/phrase -> readable text)
- **Text-to-Sign Animation** (typed/spoken text -> hand animation sequence)
- **Bilingual text workflows** (Hindi + English interface and output support)

This PRD is grounded in the current scratch project in this workspace:
- `collect-data.py` (dataset capture)
- `model.py` / `sign_model.pkl` (training + classification artifact)
- `sign-detection.py` (real-time webcam detection)

## 2) Problem Statement

Communication barriers persist between hearing/speaking users and people with hearing/speech impairments. Most lightweight demos are not production-ready, do not support Indian context/languages adequately, and lack polished, accessible UI/UX.

## 3) Goals and Non-Goals

### Goals
- Deliver a **minimalist, pastel, non-neon** web experience.
- Support **real-time webcam-based ISL sign recognition**.
- Support **text-to-sign hand animation playback**.
- Provide **Hindi and English** text handling for communication workflows.
- Be deployable and usable on **desktop + mobile browsers**.
- Offer robust frontend, backend, and API boundaries for future scaling.

### Non-Goals (Phase 1)
- Full sentence-level grammar-perfect ISL generation from arbitrary paragraphs.
- Native offline app packaging (PWA support is in scope, native app stores are not).
- Support for every regional sign dialect in first release.

## 4) Target Users

- Deaf / hard-of-hearing users.
- Family members, teachers, hospital staff, public-service counters.
- Students and beginners learning common ISL signs.

## 5) Success Metrics

### Product KPIs
- Median sign-to-text inference latency: **< 250 ms/frame** (frontend + backend path).
- End-to-end prediction update latency in UI: **< 500 ms**.
- Text-to-animation first-frame latency: **< 1.5 s**.
- Session crash rate: **< 1%**.
- 7-day retention for pilot users: **>= 25%**.

### Model KPIs
- Top-1 accuracy on curated validation set: **>= 85%** (Phase 1 vocabulary).
- False positive rate on idle/no-hand scenes: **< 8%**.

## 6) Functional Requirements

### FR-1 Sign-to-Text (Webcam)
- User can start webcam from browser permission prompt.
- User sees mirrored live preview + hand landmarks overlay (toggle).
- System predicts sign class and confidence continuously.
- Temporal smoothing is applied before final text commit.
- Output panel shows:
  - current top prediction
  - confidence
  - rolling sentence buffer
- User can clear/copy/export transcript.

### FR-2 Text-to-Sign Animation
- User enters text in Hindi or English.
- System maps tokens/phrases to animation units.
- Playback controls: play, pause, speed, replay, step.
- Missing token fallback: spell mode / closest mapped phrase / warning.
- Paper-style subtle animation aesthetic.

### FR-3 Language and Localization
- UI language switch: English <-> Hindi.
- Input text normalization pipeline for both languages.
- Output text shown in selected UI language where applicable.

### FR-4 Conversation Mode
- Split view:
  - Left: live sign-to-text stream
  - Right: text input and animation playback
- Optimized for fast turn-taking in real conversations.

### FR-5 Accessibility and Inclusivity
- Keyboard navigable controls.
- Screen-reader compatible labels and ARIA tags.
- High-contrast mode toggle while preserving palette family.
- Captions and visual state indicators for all key interactions.

### FR-6 Admin/Model Ops (Basic)
- Versioned model loading endpoint.
- Health endpoint for webcam inference service.
- Ability to switch active model by config (no code redeploy).

## 7) Non-Functional Requirements

### Performance
- FPS target on mid-range laptop camera pipeline: **>= 18 FPS effective UI update**.
- API p95 response (inference endpoint): **< 300 ms** under baseline load.

### Reliability
- Graceful degradation if camera unavailable.
- Retry-safe API behavior and explicit error codes.

### Security & Privacy
- HTTPS mandatory in production.
- Explicit consent before webcam activation.
- No raw frame retention by default.
- Optional anonymized telemetry only with consent.

### Cross-Platform
- Browsers: Chrome, Edge, Firefox, Safari (latest stable -1).
- Devices: Windows, macOS, Linux, Android, iOS browsers.

## 8) UX and Visual Design Requirements

### Visual Direction
- **Pure Two-Color Dual-Inversion Style:** An extremely clean, scholarly, Swiss-style minimalist visual system that is exceptionally structured, serious, and professional.
- **Strictly NO neon glows, saturated accents, or childish elements.** All borders are fine, solid geometric lines. Theme toggling shifts colors through a dual-inversion of the two main colors.

### Mandatory Palette
- **Primary Pale Grey:** `#EBEBDF`
- **Accent Rich Blue:** `#191265`
- **Light Theme:** Background is strictly Pale Grey (`#EBEBDF`), with typography, frames, boundaries, and interactive shapes rendered in Rich Blue (`#191265`).
- **Dark Theme (Pure Inversion):** Background is strictly Rich Blue (`#191265`), with typography, frames, boundaries, and interactive shapes rendered in Pale Grey (`#EBEBDF`).

### Typography
- **Headings & Brand:** **Peace Sans** (mapped to Space Grotesk / geometric authoritative typeface)
- **Body & UI Text:** **Open Sauce One** (mapped to Plus Jakarta Sans / Inter clean geometric typography)

### Motion
- Paper-style, subtle geometric easing.
- Soft page flips and structural layer transitions. No high-frequency or flashy consumer effects.

## 9) Core Screens

- **Landing Page:** Professional hero with structured typographic grids, features table/cards, live video preview, testimonials, and structural footer.
- **Live Translator Page:** Real-time MediaPipe hand landmarks tracking camera feed with a clean, low-opacity drawing overlay, translation confidence, rolling sentence transcripts, and voice output (TTS). Includes a Light/Dark theme toggle.
- **Text-to-Sign Page:** User types text and views sequential card-based hand sign playback and a clean academic-style hand gesture avatar executing mapped gestures.
- **Dashboard:** Translation history logs, saved phrase cards, usage metrics rendered in beautiful custom SVG charts, and custom gesture trainer.
- **Login/Register Page:** Minimal, structured SaaS login/register card interface with simulated JWT secure authorization.
- **About Project Page:** Interactive storytelling timeline showing the research scope, the tech stack (Flask backend + React client), and the academic team.


## 10) API-Level Product Requirements

- `POST /api/v1/infer/frame` for sign prediction from frame/landmark payload.
- `POST /api/v1/text-to-sign` for animation token sequence generation.
- `GET /api/v1/animations/:token` for animation asset metadata.
- `GET /api/v1/health` for service readiness.
- `GET /api/v1/models/active` for diagnostics.

## 11) Release Plan

### Phase 1 (MVP)
- Limited vocabulary sign classifier (based on current scratch model classes + expanded curated set)
- Real-time sign-to-text UI
- Text-to-sign animation for core vocabulary
- English/Hindi UI toggle

### Phase 2
- Larger vocabulary and sentence smoothing
- Better ambiguity resolution
- PWA enhancements and low-bandwidth optimizations

### Phase 3
- Domain packs (healthcare, education, public service phrases)
- Optional on-device inference mode

## 12) Risks and Mitigations

- Dataset bias / limited vocabulary -> curated balanced dataset expansion + confidence gating.
- Webcam permission denial -> manual upload/test mode fallback.
- Hindi tokenization complexity -> normalized transliteration and phrase dictionary layer.
- Animation coverage gaps -> fallback spelling mode + content roadmap.

## 13) Open Decisions

- Confirm Hindi output behavior expectations (translation vs direct transcription for sign labels).
- Confirm target vocabulary size for MVP.

## 14) Acceptance Criteria (MVP)

- User can open site, grant camera, and see live prediction stream.
- User can type Hindi/English text and view hand animation playback.
- Design matches pastel minimal spec, with required typography and no neon styling.
- App works across major desktop/mobile browsers with no blocking errors.
- Backend APIs documented and testable with stable response contracts.

