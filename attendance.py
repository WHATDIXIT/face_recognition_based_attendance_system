"""
attendance.py
=============
Real-time face recognition engine.
Matches webcam frames against stored encodings and marks attendance.
"""

import logging
import time
from typing import Generator

import cv2
import numpy as np

from database      import mark_attendance
from face_encoder  import load_encodings, FR_AVAILABLE, encodings_exist

logger = logging.getLogger(__name__)

# ── Tuning knobs ─────────────────────────────────────────────────────────────
TOLERANCE        = 0.52   # lower → stricter match
RECOGNITION_WAIT = 3.0    # seconds between re-marking the same face


# ════════════════════════════════════════════════════════════════════════════
#  Face recognition helper
# ════════════════════════════════════════════════════════════════════════════

class FaceRecognitionEngine:
    """
    Wraps face_recognition + OpenCV into a clean, reusable object.
    Maintains a cooldown dict to prevent duplicate rapid markings.
    """

    def __init__(self):
        self._cooldown: dict[str, float] = {}   # roll_no → last_marked_time
        self._data     = {"encodings": [], "names": [], "rolls": []}
        self._cascade  = self._load_cascade()
        self.reload_encodings()

    # ── Setup ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_cascade() -> cv2.CascadeClassifier:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cc   = cv2.CascadeClassifier(path)
        return cc

    def reload_encodings(self) -> None:
        self._data = load_encodings()
        logger.info(
            "Loaded %d face encodings for %d students.",
            len(self._data["encodings"]),
            len(set(self._data["rolls"])),
        )

    # ── Per-frame processing ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Detect and recognise faces in a BGR frame.

        Returns
        -------
        annotated_frame : BGR ndarray with drawn boxes & labels
        results         : list of dicts with recognition info
        """
        if not FR_AVAILABLE:
            cv2.putText(frame, "face_recognition not installed",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, []

        import face_recognition

        # Downscale for speed
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        results = []
        for enc, (top, right, bottom, left) in zip(face_encodings, face_locations):
            name       = "Unknown"
            roll_no    = ""
            confidence = 0.0

            if self._data["encodings"]:
                matches   = face_recognition.compare_faces(
                    self._data["encodings"], enc, tolerance=TOLERANCE
                )
                distances = face_recognition.face_distance(
                    self._data["encodings"], enc
                )

                best_idx = int(np.argmin(distances))
                if matches[best_idx]:
                    name       = self._data["names"][best_idx]
                    roll_no    = self._data["rolls"][best_idx]
                    confidence = round((1 - distances[best_idx]) * 100, 1)

            # Scale coords back to original size
            top    *= 2; right *= 2; bottom *= 2; left *= 2

            # Draw bounding box
            color = (0, 229, 255) if roll_no else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Label background
            label      = f"{name}  {roll_no}  {confidence:.1f}%" if roll_no else "Unknown"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            cv2.rectangle(frame,
                          (left, bottom - 22),
                          (left + label_size[0] + 8, bottom),
                          color, cv2.FILLED)
            cv2.putText(frame, label,
                        (left + 4, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

            # Mark attendance (with cooldown)
            if roll_no:
                now = time.time()
                if now - self._cooldown.get(roll_no, 0) >= RECOGNITION_WAIT:
                    res = mark_attendance(roll_no, name)
                    self._cooldown[roll_no] = now
                    results.append({
                        "name":       name,
                        "roll_no":    roll_no,
                        "confidence": confidence,
                        "marked":     res["new_entry"],
                        "message":    res["message"],
                    })
                else:
                    results.append({
                        "name":       name,
                        "roll_no":    roll_no,
                        "confidence": confidence,
                        "marked":     False,
                        "message":    "Cooldown active.",
                    })
            else:
                results.append({
                    "name":       "Unknown",
                    "roll_no":    "",
                    "confidence": confidence,
                    "marked":     False,
                    "message":    "Face not recognised.",
                })

        return frame, results


# ════════════════════════════════════════════════════════════════════════════
#  Streamlit-friendly generator (yield frames as JPEG bytes)
# ════════════════════════════════════════════════════════════════════════════

def run_recognition_session(
    max_seconds: int = 30,
    camera_index: int = 0,
) -> Generator[tuple[bytes, list[dict]], None, None]:
    """
    Open webcam and yield (jpeg_bytes, results_list) per frame for `max_seconds`.
    Designed to be consumed inside a Streamlit st.empty() loop.
    """
    engine = FaceRecognitionEngine()
    cap    = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        yield b"", [{"message": "Cannot open webcam."}]
        return

    deadline = time.time() + max_seconds
    try:
        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                break
            annotated, results = engine.process_frame(frame)
            _, jpeg = cv2.imencode(".jpg", annotated)
            yield jpeg.tobytes(), results
    finally:
        cap.release()


# ════════════════════════════════════════════════════════════════════════════
#  Single-image recognition (for uploaded images)
# ════════════════════════════════════════════════════════════════════════════

def recognise_from_image_bytes(image_bytes: bytes) -> tuple[bytes, list[dict]]:
    """
    Recognise faces in a single image (bytes).
    Returns (annotated_jpeg_bytes, results_list).
    """
    engine  = FaceRecognitionEngine()
    nparr   = np.frombuffer(image_bytes, np.uint8)
    frame   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return b"", [{"message": "Invalid image."}]

    annotated, results = engine.process_frame(frame)
    _, jpeg = cv2.imencode(".jpg", annotated)
    return jpeg.tobytes(), results
