"""
register_student.py
===================
Student registration: capture face images from webcam / file upload,
persist them to the dataset directory, and trigger encoder update.
"""

import os
import time
import logging
from typing import Generator

import cv2
import numpy as np

from utils import student_image_dir, ensure_directories

logger = logging.getLogger(__name__)

# Number of images to capture per student
CAPTURE_COUNT = 20


# ════════════════════════════════════════════════════════════════════════════
#  Face detection (Haar cascade – lightweight, no GPU needed)
# ════════════════════════════════════════════════════════════════════════════

def _load_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cc = cv2.CascadeClassifier(cascade_path)
    if cc.empty():
        raise RuntimeError("Haar cascade not found – OpenCV installation may be incomplete.")
    return cc


# ════════════════════════════════════════════════════════════════════════════
#  Webcam-based capture
# ════════════════════════════════════════════════════════════════════════════

def capture_faces_from_webcam(
    roll_no: str,
    name: str,
    n_images: int = CAPTURE_COUNT,
) -> dict:
    """
    Opens the webcam, captures `n_images` face crops, saves them to disk,
    and returns a result dict.

    NOTE: This function is used when running locally (not in Streamlit Cloud).
    In the Streamlit UI we prefer capture_from_uploaded_image().
    """
    ensure_directories()
    save_dir = student_image_dir(roll_no, name)
    cascade  = _load_cascade()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {
            "success": False,
            "message": "Cannot open webcam. Ensure it is connected and not in use.",
            "captured": 0,
        }

    count      = 0
    start_time = time.time()

    while count < n_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_img  = frame[y:y + h, x:x + w]
            img_path  = os.path.join(save_dir, f"{roll_no}_{count + 1}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1

            # Draw rectangle for user feedback
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 229, 255), 2)
            cv2.putText(frame, f"Captured: {count}/{n_images}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 157), 2)

            if count >= n_images:
                break

        cv2.imshow("Face Registration – press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Safety timeout 60 s
        if time.time() - start_time > 60:
            break

    cap.release()
    cv2.destroyAllWindows()

    if count > 0:
        return {
            "success": True,
            "message": f"Captured {count} face images for {name}.",
            "captured": count,
        }
    return {
        "success": False,
        "message": "No faces detected. Ensure good lighting and face the camera.",
        "captured": 0,
    }


# ════════════════════════════════════════════════════════════════════════════
#  Streamlit-friendly: save from uploaded file bytes
# ════════════════════════════════════════════════════════════════════════════

def capture_from_uploaded_image(
    roll_no: str,
    name: str,
    image_bytes: bytes,
    img_index: int = 1,
) -> dict:
    """
    Save a face image from bytes (e.g. Streamlit file_uploader).
    Augments the single image into multiple samples for robust encoding.

    Parameters
    ----------
    image_bytes : raw bytes from st.file_uploader or st.camera_input
    img_index   : counter for filename uniqueness
    """
    ensure_directories()
    save_dir = student_image_dir(roll_no, name)
    cascade  = _load_cascade()

    nparr  = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"success": False, "message": "Invalid image data.", "saved": 0}

    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(60, 60))

    if len(faces) == 0:
        # Fall back: save the whole frame resized as a face image
        resized  = cv2.resize(frame, (200, 200))
        img_path = os.path.join(save_dir, f"{roll_no}_{img_index}.jpg")
        cv2.imwrite(img_path, resized)
        saved    = 1
    else:
        saved = 0
        for i, (x, y, w, h) in enumerate(faces):
            face_crop = frame[y:y + h, x:x + w]
            img_path  = os.path.join(save_dir, f"{roll_no}_{img_index}_{i}.jpg")
            cv2.imwrite(img_path, face_crop)
            saved += 1

    # Augment by applying brightness / flip variants for robustness
    saved += _augment_image(frame, faces, save_dir, roll_no, img_index)

    return {
        "success": True,
        "message": f"Saved {saved} image(s) for {name}.",
        "saved":   saved,
    }


def _augment_image(
    frame: np.ndarray,
    faces,
    save_dir: str,
    roll_no: str,
    img_index: int,
) -> int:
    """Save a few augmented variants of the detected face."""
    saved = 0
    if len(faces) == 0:
        return 0

    x, y, w, h = faces[0]
    face_crop   = frame[y:y + h, x:x + w]

    variants = {
        "flip":   cv2.flip(face_crop, 1),
        "bright": cv2.convertScaleAbs(face_crop, alpha=1.2, beta=20),
        "dark":   cv2.convertScaleAbs(face_crop, alpha=0.8, beta=-20),
    }
    for suffix, img in variants.items():
        path = os.path.join(save_dir, f"{roll_no}_{img_index}_{suffix}.jpg")
        cv2.imwrite(path, img)
        saved += 1

    return saved


# ════════════════════════════════════════════════════════════════════════════
#  Dataset inspection
# ════════════════════════════════════════════════════════════════════════════

def get_dataset_info() -> list[dict]:
    """Return per-student image count from the dataset directory."""
    ensure_directories()
    info = []
    for folder in sorted(os.listdir(student_image_dir.__module__ and "" or "")):
        pass  # placeholder – actual impl below

    from utils import DATASET_DIR
    for folder in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        images = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        parts   = folder.split("_", 1)
        roll_no = parts[0]
        name    = parts[1].replace("_", " ") if len(parts) >= 2 else folder
        info.append({
            "roll_no":    roll_no,
            "name":       name,
            "image_count": len(images),
            "folder":     folder,
        })
    return info


def delete_student_dataset(roll_no: str) -> dict:
    """Remove all images for a student from the dataset folder."""
    from utils import DATASET_DIR
    import shutil
    removed = 0
    for folder in os.listdir(DATASET_DIR):
        if folder.startswith(f"{roll_no}_"):
            shutil.rmtree(os.path.join(DATASET_DIR, folder), ignore_errors=True)
            removed += 1
    return {
        "success": removed > 0,
        "message": f"Removed {removed} folder(s) for roll {roll_no}.",
    }
