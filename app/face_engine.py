import numpy as np
import cv2
import face_recognition

def bytes_to_ndarray(jpeg_bytes: bytes):
    data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def compute_encodings_from_bgr(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return []
    encs = face_recognition.face_encodings(rgb, boxes)
    return encs

def average_encoding(encodings):
    if not encodings:
        return None
    arr = np.array(encodings)
    return arr.mean(axis=0)

def match_encoding(encoding, known_encodings, tolerance=0.45):
    # Returns index of best match or -1
    if not known_encodings:
        return -1, None
    dists = face_recognition.face_distance(np.array(known_encodings), encoding)
    idx = int(np.argmin(dists))
    if dists[idx] <= tolerance:
        return idx, float(dists[idx])
    return -1, float(dists[idx])



# --- Blink detection helpers (optional, uses dlib shape predictor) ---
def eye_aspect_ratio(eye):
    # eye: array-like of 6 (x,y) points
    import numpy as np
    A = ((eye[1][0]-eye[5][0])**2 + (eye[1][1]-eye[5][1])**2) ** 0.5
    B = ((eye[2][0]-eye[4][0])**2 + (eye[2][1]-eye[4][1])**2) ** 0.5
    C = ((eye[0][0]-eye[3][0])**2 + (eye[0][1]-eye[3][1])**2) ** 0.5
    if C == 0: return 0.0
    return (A + B) / (2.0 * C)

def detect_blinks_bounding(img_bgr, predictor_path=None, threshold=0.20):
    """Detect whether eyes are blinking (returns True if blink detected).
    Requires dlib and a facial landmarks predictor. If predictor_path is None or not found,
    this function returns None (unsupported).
    """
    try:
        import dlib
        import numpy as np
    except Exception:
        return None

    if predictor_path is None:
        # look for default in data/
        predictor_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(predictor_path):
        return None

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if not rects:
        return False
    # indices for left and right eye in 68-point model
    LEFT_EYE = list(range(36, 42))
    RIGHT_EYE = list(range(42, 48))

    for r in rects:
        shape = predictor(gray, r)
        coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        left = [coords[i] for i in LEFT_EYE]
        right = [coords[i] for i in RIGHT_EYE]
        lar = eye_aspect_ratio(left)
        rar = eye_aspect_ratio(right)
        if (lar + rar) / 2.0 < threshold:
            return True
    return False
