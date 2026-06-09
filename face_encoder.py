"""
face_encoder.py
===============
Generates and persists face encodings from the dataset folder.
Uses the face_recognition library (dlib back-end).
"""

import os
import pickle
import logging
from typing import Any

import cv2
import numpy as np

from utils import DATASET_DIR, ENCODINGS_FILE, ensure_directories

# Try importing face_recognition; graceful fallback if not installed
try:
    import face_recognition
    FR_AVAILABLE = True
except ImportError:
    FR_AVAILABLE = False
    logging.warning("face_recognition not installed – encoding disabled.")


logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Encoding generation
# ════════════════════════════════════════════════════════════════════════════

def generate_encodings(progress_callback=None) -> dict[str, Any]:
    """
    Walk the dataset directory, encode every face image, and save results
    to a pickle file.

    Returns
    -------
    {
        "success"      : bool,
        "message"      : str,
        "total_images" : int,
        "total_students": int,
    }
    """
    ensure_directories()

    if not FR_AVAILABLE:
        return {
            "success": False,
            "message": "face_recognition library is not installed.",
            "total_images": 0,
            "total_students": 0,
        }

    known_encodings: list[np.ndarray] = []
    known_names:     list[str]        = []
    known_rolls:     list[str]        = []

    student_dirs = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    total_images  = 0
    skipped       = 0

    for idx, folder_name in enumerate(student_dirs):
        folder_path = os.path.join(DATASET_DIR, folder_name)

        # Parse roll_no and name from folder name pattern "ROLL_Name"
        parts   = folder_name.split("_", 1)
        roll_no = parts[0] if len(parts) >= 1 else folder_name
        name    = parts[1].replace("_", " ") if len(parts) >= 2 else folder_name

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                img_bgr  = cv2.imread(img_path)
                if img_bgr is None:
                    skipped += 1
                    continue
                img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                encs     = face_recognition.face_encodings(img_rgb)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(name)
                    known_rolls.append(roll_no)
                    total_images += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.error("Error encoding %s: %s", img_path, e)
                skipped += 1

        if progress_callback:
            progress_callback((idx + 1) / max(len(student_dirs), 1))

    # Persist
    data = {
        "encodings": known_encodings,
        "names":     known_names,
        "rolls":     known_rolls,
    }
    os.makedirs(os.path.dirname(ENCODINGS_FILE), exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    return {
        "success":        True,
        "message":        (
            f"Encodings generated: {total_images} faces from "
            f"{len(student_dirs)} students ({skipped} skipped)."
        ),
        "total_images":   total_images,
        "total_students": len(student_dirs),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Encoding loading
# ════════════════════════════════════════════════════════════════════════════

def load_encodings() -> dict[str, Any]:
    """
    Load persisted encodings from pickle.

    Returns
    -------
    {"encodings": list, "names": list, "rolls": list}
    or empty lists if the file is missing.
    """
    if not os.path.exists(ENCODINGS_FILE):
        return {"encodings": [], "names": [], "rolls": []}

    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        logger.error("Failed to load encodings: %s", e)
        return {"encodings": [], "names": [], "rolls": []}


def encodings_exist() -> bool:
    return os.path.exists(ENCODINGS_FILE) and os.path.getsize(ENCODINGS_FILE) > 0
