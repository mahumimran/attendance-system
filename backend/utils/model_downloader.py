"""
Downloads the YuNet (face detection) and SFace (face embedding) ONNX models
from OpenCV Zoo on first run, if they aren't already present locally.

Both models are small (~2-40 MB combined), CPU-only, and run through
OpenCV's own DNN module — no dlib, no CMake, no C++ compiler, and no
AVX-instruction requirement, which makes them work on older/budget CPUs
that crash on prebuilt dlib wheels.
"""
import urllib.request

from backend.utils.config import settings


def _download(url: str, destination) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"[models] Downloading {destination.name} ...")
    urllib.request.urlretrieve(url, str(destination))
    print(f"[models] Saved to {destination}")


def ensure_models_downloaded() -> None:
    """Called once on app startup. Safe to call repeatedly — skips existing files."""
    if not settings.YUNET_MODEL_PATH.exists():
        _download(settings.YUNET_MODEL_URL, settings.YUNET_MODEL_PATH)

    if not settings.SFACE_MODEL_PATH.exists():
        _download(settings.SFACE_MODEL_URL, settings.SFACE_MODEL_PATH)
