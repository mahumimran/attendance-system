"""
Face recognition service — OpenCV DNN backend (YuNet + SFace).

Pipeline implemented in this file:

    Frame (base64 JPEG from browser)
        -> decode to OpenCV BGR image
        -> Face Detection            (YuNet - a small CNN detector, ONNX,
                                       runs through OpenCV's own DNN module)
        -> Quality checks            (exactly one face, min size, sharpness)
        -> Face Alignment            (SFace's alignCrop(), which warps the
                                       face using the 5 landmarks YuNet returns)
        -> Embedding Generation      (SFace - 128-d embedding, ONNX)
        -> Similarity Comparison     (cosine similarity vs stored embeddings)
        -> Threshold Validation      (settings.FACE_MATCH_THRESHOLD)
        -> Student Identification    (best match above threshold, else "unknown")

Why OpenCV DNN instead of dlib:
Both YuNet and SFace are plain ONNX models executed through OpenCV's own
"dnn" module, which dispatches to whatever SIMD instructions the current
CPU actually supports at runtime. Prebuilt dlib wheels, by contrast, are
statically compiled with a fixed instruction set (often AVX/AVX2) baked in
at build time -- on a CPU that lacks those instructions, importing dlib
crashes with an illegal-instruction error instead of falling back
gracefully. Since this project already depends on opencv-python for image
handling, using OpenCV's own face models avoids a second, less portable
native dependency entirely.

Why embeddings instead of raw image comparison:
Raw pixel/image comparison breaks under lighting changes, pose, expression,
and camera differences. An embedding model instead maps a face to a point
in a 128-dimensional space such that images of the *same* person land close
together (by cosine similarity) and different people land far apart,
regardless of superficial pixel differences.
"""
import base64
import io
import json
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image

from backend.utils.config import settings

_detector = None
_recognizer = None


class FaceQualityError(Exception):
    """Raised when a captured frame fails detection/quality checks."""


def _get_detector():
    """Lazily creates the YuNet face detector (loaded once, reused)."""
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            str(settings.YUNET_MODEL_PATH),
            "",
            (320, 320),
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=50,
        )
    return _detector


def _get_recognizer():
    """Lazily creates the SFace embedding model (loaded once, reused)."""
    global _recognizer
    if _recognizer is None:
        _recognizer = cv2.FaceRecognizerSF.create(str(settings.SFACE_MODEL_PATH), "")
    return _recognizer


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Convert a base64 data-URL or raw base64 string into a BGR numpy array (OpenCV format)."""
    if "," in image_base64 and image_base64.strip().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    img_bytes = base64.b64decode(image_base64)
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def _sharpness_score(bgr_image: np.ndarray) -> float:
    """Variance of the Laplacian -- a standard, cheap blur/quality proxy."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_and_validate_single_face(bgr_image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Runs YuNet face detection and quality gating.

    Returns (face_row, sharpness_score), where face_row is the 1x15 detection
    row [x, y, w, h, 5x(landmark_x, landmark_y), confidence] that SFace needs
    for alignment.

    Raises FaceQualityError with a user-friendly message on any failure,
    covering: no face, multiple faces, face too small, image too blurry.
    """
    detector = _get_detector()
    h, w = bgr_image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr_image)

    if faces is None or len(faces) == 0:
        raise FaceQualityError("No face detected. Please face the camera directly.")
    if len(faces) > 1:
        raise FaceQualityError(
            "Multiple faces detected. Only one person should be in frame."
        )

    face_row = faces[0]
    face_width, face_height = face_row[2], face_row[3]
    if min(face_width, face_height) < settings.MIN_FACE_SIZE_PX:
        raise FaceQualityError("Face is too small/far. Please move closer to the camera.")

    sharpness = _sharpness_score(bgr_image)
    if sharpness < settings.MIN_IMAGE_SHARPNESS:
        raise FaceQualityError("Image is too blurry. Please hold steady and ensure good lighting.")

    return face_row, sharpness


def generate_embedding(bgr_image: np.ndarray, face_row: np.ndarray) -> np.ndarray:
    """
    Face Alignment + Embedding Generation.

    SFace's alignCrop() warps the detected face using the 5 landmarks YuNet
    returned (eyes, nose, mouth corners) before feature() runs it through
    the embedding network, producing a 128-d vector.
    """
    recognizer = _get_recognizer()
    aligned_face = recognizer.alignCrop(bgr_image, face_row)
    feature = recognizer.feature(aligned_face)
    return feature.flatten().astype(np.float64)


def embedding_to_json(embedding: np.ndarray) -> str:
    return json.dumps(embedding.tolist())


def embedding_from_json(embedding_json: str) -> np.ndarray:
    return np.array(json.loads(embedding_json), dtype=np.float32)


def process_registration_frame(image_base64: str) -> Tuple[str, float]:
    """
    Full pipeline for student face registration.
    Returns (embedding_json, quality_score) or raises FaceQualityError.
    """
    bgr_image = decode_base64_image(image_base64)
    face_row, sharpness = detect_and_validate_single_face(bgr_image)
    embedding = generate_embedding(bgr_image, face_row)
    return embedding_to_json(embedding), sharpness


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Same metric SFace's own .match(..., FaceRecognizerSF_FR_COSINE) uses."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def find_best_match(
    probe_embedding: np.ndarray,
    known_embeddings: List[np.ndarray],
    known_ids: List[int],
) -> Optional[Tuple[int, float]]:
    """
    Similarity Comparison + Threshold Validation.

    Computes cosine similarity between the probe embedding and every
    registered embedding -- higher means more similar. The closest match is
    accepted only if its similarity is at or above FACE_MATCH_THRESHOLD;
    otherwise the face is treated as unknown. This threshold is what
    prevents "close but not confident" matches from silently marking the
    wrong student present.
    """
    if not known_embeddings:
        return None

    scores = [_cosine_similarity(probe_embedding, known) for known in known_embeddings]
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    if best_score >= settings.FACE_MATCH_THRESHOLD:
        return known_ids[best_idx], best_score
    return None


def similarity_to_confidence(score: float) -> float:
    """
    Converts a cosine-similarity score into an intuitive 0-100% confidence,
    scaled so the match threshold itself sits at 0% and a perfect match
    (score of 1.0) sits at 100%.
    """
    span = max(1.0 - settings.FACE_MATCH_THRESHOLD, 1e-6)
    confidence = max(0.0, (score - settings.FACE_MATCH_THRESHOLD) / span) * 100
    return round(min(confidence, 99.9), 1)


def process_recognition_frame(
    image_base64: str,
    known_embeddings: List[np.ndarray],
    known_ids: List[int],
) -> dict:
    """
    Full pipeline for live attendance recognition.
    Returns a dict describing the outcome; never raises for "unknown face" --
    only for genuinely bad frames (no face / multiple faces / low quality),
    which the route layer turns into user-friendly error responses.
    """
    bgr_image = decode_base64_image(image_base64)
    face_row, _ = detect_and_validate_single_face(bgr_image)
    probe_embedding = generate_embedding(bgr_image, face_row)

    match = find_best_match(probe_embedding, known_embeddings, known_ids)
    if match is None:
        return {"matched": False}

    student_pk, score = match
    return {
        "matched": True,
        "student_pk": student_pk,
        "confidence": similarity_to_confidence(score),
    }
