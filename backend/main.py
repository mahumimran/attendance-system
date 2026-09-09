"""
Application entrypoint.

Run with:  uvicorn backend.main:app --reload --port 8000
Frontend is served statically at "/" so the whole app is reachable from a
single origin (avoids CORS friction during local development).
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.database import init_db
from backend.routes import students, attendance, reports
from backend.utils.config import settings
from backend.utils.model_downloader import ensure_models_downloaded

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered school attendance platform using face-embedding recognition.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    ensure_models_downloaded()
    init_db()


# ---- API routers ----
app.include_router(students.router, prefix="/api")
app.include_router(attendance.router, prefix="/api")
app.include_router(reports.router, prefix="/api")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "app": settings.APP_NAME}


# ---- Static frontend ----
# Mounted last, at the root path, so it acts as a catch-all for index.html,
# dashboard.html, css/, js/, etc. All /api/* routes above are matched first.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
