# Smart School Attendance System

An AI-powered school attendance platform combining **face-recognition computer
vision** with a **manual fallback workflow**, built on FastAPI, SQLite, and a
vanilla HTML/CSS/JS frontend.

---

## 1. Architecture Overview

```
┌─────────────────┐      REST/JSON       ┌──────────────────────┐      SQLAlchemy    ┌──────────────┐
│   Frontend       │ ───────────────────▶ │   FastAPI Backend     │ ──────────────────▶ │  SQLite DB   │
│ (HTML/CSS/JS)    │ ◀─────────────────── │  backend/main.py      │ ◀────────────────── │ attendance.db│
│  - webcam capture│                       │  routes/ (API layer)  │                     └──────────────┘
│  - dashboards    │                       │  services/ (CV logic) │
└─────────────────┘                       └──────────────────────┘
                                                     │
                                                     ▼
                                    OpenCV DNN — YuNet + SFace (ONNX)
                                    (detection, alignment, embeddings)
```

The project is split into three independent layers:

- **`backend/routes/`** — thin FastAPI endpoints that validate input (via
  Pydantic schemas) and delegate to services. No business logic lives here.
- **`backend/services/`** — the actual computer-vision pipeline
  (`face_recognition_service.py`) and attendance business rules
  (`attendance_service.py`, e.g. duplicate-prevention). These are plain
  Python and could be unit-tested independently of FastAPI.
- **`backend/models/` + `backend/database.py`** — SQLAlchemy ORM models and
  session management.
- **`frontend/`** — static HTML pages, one per screen, sharing a CSS design
  system and a small set of JS utility modules (`app.js` for API calls/toasts,
  `camera.js` for webcam handling).

This separation means the face-recognition model, the database engine, or
the frontend framework could each be swapped independently without
rewriting the others.

## 2. Why OpenCV DNN (YuNet + SFace) as the Embedding Model

The brief allowed InsightFace/ArcFace "or another suitable face-embedding
model." This project uses **YuNet** for face detection and **SFace** for
128-d face embeddings — both are small ONNX models from
[OpenCV Zoo](https://github.com/opencv/opencv_zoo), executed through
OpenCV's own `dnn` module, because:

- **No native-dependency risk.** OpenCV's `dnn` module dispatches to
  whatever SIMD instructions the CPU actually supports at runtime.
  Alternatives like `dlib` (which `face_recognition` and InsightFace's
  CPU path both lean on, directly or via similarly compiled wheels) ship
  prebuilt binaries with a fixed instruction set (often AVX/AVX2) baked in
  at compile time — on CPUs that lack those instructions, importing the
  library crashes with an illegal-instruction error rather than degrading
  gracefully. Since this project already depends on `opencv-python` for
  image handling, using OpenCV's own face models avoids a second, less
  portable native dependency entirely.
- **128-dimensional embeddings** — small, fast to store and compare.
- **Auto-downloaded, no manual setup.** The two ONNX files (~40 MB
  combined) are fetched once on first server start
  (`backend/utils/model_downloader.py`) — no CMake, no C++ compiler, no
  GPU, no separate runtime to install.
- **CPU-friendly** for a classroom laptop/teacher's PC running the
  recognition loop continuously. InsightFace's ArcFace models are more
  accurate at very large scale (thousands of identities) but typically
  benefit from GPU acceleration, which is overkill for a single
  classroom's roster.

`backend/services/face_recognition_service.py` isolates every model call
behind plain functions (`_get_detector`, `_get_recognizer`,
`generate_embedding`, `find_best_match`). Swapping in InsightFace later
means rewriting those functions only — no route, schema, or database code
changes. Instructions for that swap are in `requirements.txt`.

## 3. Database Design

**`students`**

| column          | type | notes                                                |
|-----------------|------|-------------------------------------------------------|
| id              | PK   | internal surrogate key                                |
| student_id      | str  | unique, human-facing ID (e.g. `ST-102`)               |
| name, class_name, section, roll_number | str | profile fields |
| face_embedding  | text | JSON-encoded 128-float vector, not a raw image        |
| face_registered | bool | quick flag to filter "ready for recognition" students |
| created_at      | datetime | registration timestamp |

**`attendance`**

| column          | type | notes |
|-----------------|------|-------|
| id              | PK   | |
| student_id_fk   | FK → students.id | |
| date, time      | date/time | when attendance was taken |
| status          | str  | `Present` / `Absent` |
| method          | str  | `AI Face Recognition` / `Manual` |
| confidence      | float, nullable | similarity confidence, AI method only |

**Duplicate prevention** is enforced at two levels:
1. **Application level** — `attendance_service.mark_attendance()` checks for
   an existing `(student_id, date)` row before inserting.
2. **Database level** — a `UNIQUE(student_id_fk, date)` constraint on the
   `attendance` table, which catches race conditions (e.g. two rapid
   recognition hits) that the application check might miss.

Storing the **embedding instead of the photo** keeps biometric data
minimal — you cannot reconstruct a face image from a 128-number vector,
which is a meaningfully smaller privacy footprint than storing photos.

## 4. Frontend ↔ Backend Communication

The frontend is plain static HTML/CSS/JS served by FastAPI itself
(`StaticFiles` mounted at `/`), so there's a single origin and no CORS
friction during local development. Every page loads `js/app.js`, which
exposes a small `api` object (`api.get`, `api.post`, `api.put`, `api.del`)
wrapping `fetch()` against `/api/...` endpoints and centralizing error
handling into toast notifications.

The live camera pages (`register.html`, `attendance.html`) use
`js/camera.js`'s `CameraController`, which wraps `getUserMedia`, draws
frames to an off-screen `<canvas>`, and exports them as base64 JPEG
strings — these are POSTed as JSON (`{"image_base64": "..."}`) to
`/api/students/{id}/face` (registration) or `/api/attendance/recognize`
(live attendance), which the backend decodes and feeds into the CV
pipeline.

## 5. The Face-Recognition Pipeline

```
Webcam frame (base64 JPEG)
        │
        ▼
Face Detection            — YuNet (ONNX, via cv2.FaceDetectorYN), a small
        │                    CNN detector; rejects frames with 0 or >1 faces
        ▼
Quality Validation         — face too small? image too blurry (variance-of-
        │                    Laplacian sharpness check)? both are rejected
        ▼
Face Alignment             — SFace's alignCrop() warps the face using the
        │                    5 landmarks YuNet returned (eyes, nose, mouth
        │                    corners), normalizing pose before embedding
        ▼
Embedding Generation       — SFace (ONNX) outputs a 128-dimensional vector
        │
        ▼
Similarity Comparison      — cosine similarity between the probe embedding
        │                    and every registered student's stored embedding
        ▼
Threshold Validation       — best match accepted only if similarity ≥ 0.363
        │                    (configurable via FACE_MATCH_THRESHOLD)
        ▼
Student Identification     — match → mark attendance; no match → "Unknown
                              Student", nothing is written to the database
```

**What are face embeddings, and why use them instead of raw image
comparison?** A face embedding is the output of a neural network trained
so that images of the *same* person map to nearby points in a
high-dimensional vector space, while different people map far apart.
Comparing two faces becomes a simple similarity calculation on two short
vectors, robust to lighting, angle, and expression changes — pixel-by-pixel
image comparison has none of these invariances and breaks under all of
them.

**How similarity is calculated:** Cosine similarity between two 128-d
vectors. Higher similarity = more alike faces (1.0 = identical direction).

**How the threshold works:** `FACE_MATCH_THRESHOLD` (default `0.363`, OpenCV
Zoo's published operating point for SFace) is the minimum similarity
considered a genuine match. The closest registered student is only
accepted if their similarity meets or exceeds this line; otherwise, no
match is returned. Raising the threshold makes the system stricter (fewer
false accepts, more false rejects); lowering it does the opposite.

**Why unknown faces must never be marked present:** Attendance is a legal
and academic record. A system that guesses on low-confidence matches would
silently mark the wrong student present (or a stranger as *some* student),
corrupting the record with no way to detect the error later. Rejecting
uncertain matches and surfacing "Unknown Student" keeps a human able to
fall back to Manual Attendance for edge cases.

## 6. Project Structure

```
project/
├── backend/
│   ├── main.py                    # FastAPI app, CORS, static mounting
│   ├── database.py                # SQLAlchemy engine/session/init
│   ├── models/models.py           # Student, Attendance ORM models
│   ├── models/onnx/               # Auto-downloaded YuNet + SFace models
│   ├── schemas/schemas.py         # Pydantic request/response schemas
│   ├── routes/
│   │   ├── students.py            # CRUD + face registration endpoint
│   │   ├── attendance.py          # AI recognition + manual + listing
│   │   └── reports.py             # Aggregation + CSV/Excel export
│   ├── services/
│   │   ├── face_recognition_service.py   # CV pipeline (YuNet + SFace)
│   │   └── attendance_service.py         # Duplicate-prevention logic
│   └── utils/
│       ├── config.py              # Environment-driven settings
│       └── model_downloader.py    # Fetches ONNX models on first run
├── frontend/
│   ├── index.html                 # Animated landing/login
│   ├── dashboard.html             # Stat cards + charts
│   ├── register.html              # Student form + face capture
│   ├── attendance.html            # Live AI recognition
│   ├── manual_attendance.html     # Manual entry grid
│   ├── students.html              # Student management table
│   ├── records.html               # Filterable attendance log
│   ├── reports.html               # Aggregation + export
│   ├── partials/sidebar.html      # Shared nav, injected by app.js
│   ├── css/style.css              # Design system
│   └── js/{app.js, camera.js}
├── database/attendance.db         # Created automatically on first run
├── requirements.txt
├── .env.example
└── README.md
```

## 7. Setup & Running

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies (pure Python + OpenCV, no compiler needed)
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Run the server (serves both API and frontend on one port)
uvicorn backend.main:app --reload --port 8000
```

On first start, the server automatically downloads the two small ONNX
models (~40 MB combined) it needs — this requires internet access on that
first run only; after that they're cached in `backend/models/onnx/` and
the app works fully offline.

Then open **http://localhost:8000** in your browser. Demo login:
`admin` / `changeme123` (see Security notes below — this is intentionally
simple for a course project).

**First-time workflow:**
1. Go to **Register Student**, fill in details, save, then capture a face
   with good lighting, one person in frame, facing the camera.
2. Go to **AI Face Attendance**, start the camera — attendance is marked
   automatically on a confident recognition.
3. Use **Manual Attendance** for any student without a registered face, or
   as a fallback if the camera is unavailable.

## 8. Security & Validation Notes

- All request bodies are validated with Pydantic schemas (`schemas/schemas.py`).
- No secrets are hard-coded — `ADMIN_USERNAME`, `ADMIN_PASSWORD`, and
  `SECRET_KEY` are read from `.env` via `backend/utils/config.py`.
- The current login (`index.html`) is a **client-side gate only**, suitable
  for a course project demo. For real deployment, replace it with a
  server-side session/JWT flow — the API layer is already structured so a
  `Depends(get_current_user)` dependency could be added to
  `routes/students.py` / `routes/attendance.py` without restructuring
  anything else.
- Face embeddings, not raw photos, are what's persisted — see Database
  Design above.
- Destructive actions (deleting a student) require confirmation via a
  modal dialog (`confirmModal()` in `app.js`).

## 9. Error Handling Coverage

| Scenario | Handling |
|---|---|
| Camera unavailable / permission denied | `camera.js` catches `getUserMedia` errors and shows a specific toast (`NotAllowedError`, `NotFoundError`, `NotReadableError`) |
| No face / multiple faces detected | `FaceQualityError` raised in the CV service, surfaced as a friendly message, not a stack trace |
| Low-quality / blurry frame | Laplacian-variance sharpness check rejects the frame before embedding |
| Recognition confidence too low | Distance-threshold check returns "Unknown Student" instead of guessing |
| Student already registered | `409 Conflict` from `POST /students`, backed by a DB unique constraint |
| Student already marked present | `attendance_service.mark_attendance()` returns "Attendance already recorded for today" without creating a duplicate row |
| Empty student database | `/attendance/recognize` returns a clear `400` message instead of failing on an empty comparison list |
| Database errors | SQLAlchemy `IntegrityError` caught and translated to a friendly API response |
| Backend unavailable | `app.js`'s `apiRequest()` wraps `fetch()` failures with a "Cannot reach the server" message |

## 10. Extending This Project

- **Swap the CV model**: replace the three functions in
  `face_recognition_service.py` with InsightFace's `FaceAnalysis` app +
  cosine similarity; keep everything else unchanged.
- **Real authentication**: add a `/auth/login` endpoint issuing a JWT, and
  a FastAPI dependency checked on write endpoints.
- **Multi-camera / classroom mode**: the recognition endpoint already
  accepts any single frame, so multiple classroom cameras could POST to
  the same endpoint on a schedule.
