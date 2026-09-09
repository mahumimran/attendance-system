"""
Central configuration loaded from environment variables.
Keeps secrets and tunable parameters out of source code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    # --- Database ---
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'database' / 'attendance.db'}"
    )

    # --- Face recognition tuning ---
    # Cosine-similarity threshold for the 128-d SFace embeddings.
    # Higher = stricter match. 0.363 is OpenCV Zoo's published operating point
    # for this model (yields ~99.6% accuracy on their benchmark).
    FACE_MATCH_THRESHOLD: float = float(os.getenv("FACE_MATCH_THRESHOLD", "0.363"))

    # Minimum sharpness (variance of Laplacian) to accept a frame as "good quality".
    MIN_IMAGE_SHARPNESS: float = float(os.getenv("MIN_IMAGE_SHARPNESS", "60.0"))

    # Minimum face bounding-box size (pixels) relative to frame, guards against
    # tiny/far-away faces producing unreliable embeddings.
    MIN_FACE_SIZE_PX: int = int(os.getenv("MIN_FACE_SIZE_PX", "80"))

    # --- CV models (OpenCV Zoo, CPU-only, no AVX/compiler required) ---
    MODELS_DIR: Path = BASE_DIR / "backend" / "models" / "onnx"
    YUNET_MODEL_PATH: Path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    SFACE_MODEL_PATH: Path = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
    YUNET_MODEL_URL: str = (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
        "face_detection_yunet_2023mar.onnx"
    )
    SFACE_MODEL_URL: str = (
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/"
        "face_recognition_sface_2021dec.onnx"
    )

    # --- App / CORS ---
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    APP_NAME: str = os.getenv("APP_NAME", "Smart School Attendance System")

    # --- Simple auth (structured so real auth can replace it later) ---
    ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "changeme123")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-change-in-production")


settings = Settings()
