"""
run_webcam.py
=============
Standalone live webcam attendance script.
Run this directly with:  python run_webcam.py

Press  Q  to quit.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import time
from datetime import datetime

from database      import mark_attendance, init_database
from face_encoder  import load_encodings, encodings_exist, FR_AVAILABLE
from utils         import ensure_directories

TOLERANCE        = 0.52
RECOGNITION_WAIT = 3.0    # seconds cooldown per face
CAMERA_INDEX     = 0


def main():
    ensure_directories()
    init_database()

    if not FR_AVAILABLE:
        print("[ERROR] face_recognition is not installed.")
        print("        Run:  pip install face-recognition")
        sys.exit(1)

    if not encodings_exist():
        print("[ERROR] No encodings file found.")
        print("        Generate encodings first via the Streamlit app.")
        sys.exit(1)

    import face_recognition, numpy as np

    print("[INFO] Loading encodings…")
    data = load_encodings()
    print(f"[INFO] Loaded {len(data['encodings'])} encodings.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cascade     = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cooldown    = {}
    frame_skip  = 0

    print("[INFO] Webcam opened. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_skip += 1
        if frame_skip % 2 != 0:          # process every 2nd frame
            cv2.imshow("Smart Attendance System  |  Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        small  = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs   = face_recognition.face_locations(rgb)
        encs   = face_recognition.face_encodings(rgb, locs)

        for enc, (top, right, bottom, left) in zip(encs, locs):
            top    *= 2; right *= 2; bottom *= 2; left *= 2
            name    = "Unknown"
            roll_no = ""
            conf    = 0.0

            if data["encodings"]:
                matches   = face_recognition.compare_faces(
                    data["encodings"], enc, tolerance=TOLERANCE
                )
                distances = face_recognition.face_distance(data["encodings"], enc)
                best      = int(np.argmin(distances))
                if matches[best]:
                    name    = data["names"][best]
                    roll_no = data["rolls"][best]
                    conf    = round((1 - distances[best]) * 100, 1)

            color  = (0, 229, 255) if roll_no else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label  = f"{name} ({conf:.0f}%)" if roll_no else "Unknown"
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 4, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            if roll_no:
                now = time.time()
                if now - cooldown.get(roll_no, 0) >= RECOGNITION_WAIT:
                    res = mark_attendance(roll_no, name)
                    cooldown[roll_no] = now
                    ts  = datetime.now().strftime("%H:%M:%S")
                    status = "MARKED" if res["new_entry"] else "already marked"
                    print(f"[{ts}] {name} ({roll_no}) – {status}  conf={conf}%")

        # Overlay timestamp
        ts_text = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Smart Attendance System  |  Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
