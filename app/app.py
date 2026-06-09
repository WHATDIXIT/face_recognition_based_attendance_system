from flask import Flask, render_template, request, jsonify, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

from pathlib import Path
import io, csv, datetime, json
import numpy as np

from db import add_student, get_students, log_attendance, get_attendance
from face_engine import bytes_to_ndarray, compute_encodings_from_bgr, average_encoding, match_encoding

app = Flask(__name__, template_folder="templates", static_folder="static")
# Load secret key from data/secret.key (auto-generated). Change for production.
try:
    with open(Path(__file__).resolve().parent.parent / 'data' / 'secret.key', 'r', encoding='utf-8') as sk:
        app.secret_key = sk.read().strip()
except Exception:
    app.secret_key = 'change-me-to-a-random-secret'

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

# Simple User class for Flask-Login
class User(UserMixin):
    def __init__(self, id_, username, role):
        self.id = id_
        self.username = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    try:
        u = get_user_by_username('admin') if False else None
    except Exception:
        u = None
    # we will attempt to read from DB via db_extra.get_user_by_username through username stored in session fallback
    from db_extra import get_user_by_username
    # Try to find any user with matching id
    try:
        # brute force: fetch all (small user count expected)
        with __import__('sqlite3').connect(Path(__file__).resolve().parent.parent / 'data' / 'database.sqlite') as conn:
            c = conn.cursor()
            c.execute('SELECT id, username, role FROM users WHERE id = ?', (int(user_id),))
            r = c.fetchone()
            if not r:
                return None
            return User(r[0], r[1], r[2])
    except Exception:
        return None
@app.route("/login", methods=["GET", "POST"])
def login_page():
    return render_template("login.html")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/enroll")
def enroll_page():
    return render_template("enroll.html")

@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")

@app.route("/reports")
def reports_page():
    return render_template("reports.html")

@app.post("/api/enroll")
def api_enroll():
    name = request.form.get("name", "").strip()
    roll = request.form.get("roll", "").strip()
    class_section = request.form.get("class_section", "").strip()
    if not (name and roll and class_section):
        return jsonify({"ok": False, "error": "Missing fields"}), 400

    # Expect multiple images for robustness
    files = request.files.getlist("images")
    encs = []
    for f in files:
        img = bytes_to_ndarray(f.read())
        enc = compute_encodings_from_bgr(img)
        if enc:
            encs.extend(enc)
    if not encs:
        return jsonify({"ok": False, "error": "No face detected"}), 400

    avg = average_encoding(encs)
    student_id = add_student(name, roll, class_section, avg.tolist())
    return jsonify({"ok": True, "student_id": student_id})

@app.post("/api/recognize")
def api_recognize():
    subject = request.form.get("subject")
    class_section = request.form.get("class_section")
    device_id = request.form.get("device_id")
    f = request.files.get("image")
    if not f:
        return jsonify({"ok": False, "error": "No image"}), 400
    img = bytes_to_ndarray(f.read())
    encs = compute_encodings_from_bgr(img)
    if not encs:
        return jsonify({"ok": False, "error": "No face detected"}), 200

    # Load known encodings
    students = get_students()
    known = [s["encoding"] for s in students]

    results = []
    for e in encs:
        idx, dist = match_encoding(e, known)
        if idx != -1:
            sid = students[idx]["id"]
            log_attendance(sid, subject=subject, class_section=class_section, device_id=device_id)
            results.append({"match": True, "student_id": sid, "name": students[idx]["name"], "roll": students[idx]["roll"], "distance": dist})
        else:
            results.append({"match": False, "distance": dist})
    return jsonify({"ok": True, "results": results})

@app.get("/api/students")
def api_students():
    return jsonify({"ok": True, "students": get_students()})

@app.get("/api/attendance")
def api_attendance():
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")
    class_section = request.args.get("class_section")
    subject = request.args.get("subject")
    rows = get_attendance(date_from=date_from, date_to=date_to, class_section=class_section, subject=subject)
    data = []
    for r in rows:
        data.append({
            "id": r[0], "name": r[1], "roll": r[2], "class_section": r[3], "timestamp": r[4], "subject": r[5]
        })
    return jsonify({"ok": True, "data": data})

@app.get("/api/export_csv")
def api_export_csv():
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")
    class_section = request.args.get("class_section")
    subject = request.args.get("subject")
    rows = get_attendance(date_from=date_from, date_to=date_to, class_section=class_section, subject=subject)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Name", "Roll", "Class", "Timestamp", "Subject"])
    for r in rows:
        writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5] or ""])

    mem = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    filename = f"attendance_{datetime.datetime.utcnow().date()}.csv"
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=filename)

if __name__ == "__main__":
    app.run(debug=True)


# --- Simple auth & settings (appended) ---
from werkzeug.security import generate_password_hash, check_password_hash
from db_extra import add_user_raw, get_user_by_username, load_settings, save_settings

# Ensure secret key (change for production)
app.secret_key = app.secret_key if app.secret_key else 'change-me-to-a-random-secret'

# Create default admin if missing
try:
    if not get_user_by_username('admin'):
        add_user_raw('admin', generate_password_hash('admin'), role='admin')
except Exception:
    pass

from flask import session, redirect, url_for

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'GET':
        return render_template('login.html')
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    user = get_user_by_username(username)
    if not user or not check_password_hash(user['password_hash'], password):
        return render_template('login.html', error='Invalid credentials')
    user_obj = User(user['id'], user['username'], user['role'])
    login_user(user_obj)
    session['user'] = {'id': user['id'], 'username': user['username'], 'role': user['role']}
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    try:
        logout_user()
    except Exception:
        pass
    return redirect(url_for('index'))

@app.route('/settings', methods=['GET', 'POST'])
def settings_page():
    user = session.get('user')
    if not user or user.get('role') != 'admin':
        return redirect(url_for('login_page'))
    settings = load_settings()
    if request.method == 'POST':
        try:
            t = float(request.form.get('tolerance', settings.get('tolerance', 0.45)))
            settings['tolerance'] = t
            settings['last_updated'] = datetime.datetime.utcnow().isoformat()
            save_settings(settings)
        except Exception:
            pass
        return redirect(url_for('settings_page'))
    return render_template('settings.html', settings=settings)


@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    # Only allow logged in users to change their own password
    if request.method == 'GET':
        return render_template('change_password.html')
    current = request.form.get('current_password', '').strip()
    new = request.form.get('new_password', '').strip()
    confirm = request.form.get('confirm_password', '').strip()
    if not (current and new and confirm):
        return render_template('change_password.html', error='Fill all fields')
    if new != confirm:
        return render_template('change_password.html', error='New passwords do not match')
    # verify current password
    # verify current password
    user = get_user_by_username(current_user.username)

    if not user or not check_password_hash(user['password_hash'], current):
        return render_template('change_password.html', error='Current password incorrect')

    # update password in DB
    try:
        with __import__('sqlite3').connect(
                Path(__file__).resolve().parent.parent / 'data' / 'database.sqlite'
        ) as conn:
            c = conn.cursor()
            c.execute(
                'UPDATE users SET password_hash = ? WHERE id = ?',
                (generate_password_hash(new), user['id'])
            )
            conn.commit()
    except Exception as e:
        return render_template('change_password.html', error='Could not update password')

    return render_template('change_password.html', success='Password changed successfully')


@app.post("/api/auto_scan")
def api_auto_scan():
    """
    Accepts one image (group photo). Detects faces, matches them against known encodings,
    logs attendance for unique student IDs and returns list of matched students.
    """
    f = request.files.get("image")
    if not f:
        return jsonify({"ok": False, "error": "No image"}), 400
    img = bytes_to_ndarray(f.read())
    # detect faces and encodings
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return jsonify({"ok": True, "matches": [], "note": "no_faces"})
    encs = face_recognition.face_encodings(rgb, boxes)
    students = get_students()
    known = [s["encoding"] for s in students]
    unique_marked = set()
    matches = []
    settings = load_settings()
    tolerance = float(settings.get('tolerance', 0.45))
    for e in encs:
        idx, dist = match_encoding(e, known, tolerance=tolerance)
        if idx != -1:
            sid = students[idx]["id"]
            if sid in unique_marked:
                continue
            log_attendance(sid, subject=request.form.get('subject'), class_section=request.form.get('class_section'), device_id=request.form.get('device_id'))
            unique_marked.add(sid)
            matches.append({"student_id": sid, "name": students[idx]["name"], "roll": students[idx]["roll"], "distance": dist})
    return jsonify({"ok": True, "matches": matches})