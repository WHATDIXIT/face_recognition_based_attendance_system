import sqlite3
from contextlib import contextmanager
from pathlib import Path
import json
import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "database.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll TEXT NOT NULL UNIQUE,
            class_section TEXT NOT NULL,
            encoding_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            subject TEXT,
            class_section TEXT,
            device_id TEXT,
            FOREIGN KEY(student_id) REFERENCES students(id)
        )
        """)
        conn.commit()

_init_db()

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def add_student(name, roll, class_section, encoding_vec):
    enc_json = json.dumps(list(map(float, encoding_vec)))
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO students (name, roll, class_section, encoding_json, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (name, roll, class_section, enc_json, datetime.datetime.utcnow().isoformat()))
        conn.commit()
        return c.lastrowid

def get_students():
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT id, name, roll, class_section, encoding_json FROM students ORDER BY id DESC")
        rows = c.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0], "name": r[1], "roll": r[2], "class_section": r[3],
                "encoding": json.loads(r[4])
            })
        return result

def log_attendance(student_id, subject=None, class_section=None, device_id=None, timestamp=None):
    ts = timestamp or datetime.datetime.utcnow().isoformat()
    date = ts.split("T")[0]
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO attendance (student_id, timestamp, date, subject, class_section, device_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (student_id, ts, date, subject, class_section, device_id))
        conn.commit()
        return c.lastrowid

def get_attendance(date_from=None, date_to=None, class_section=None, subject=None):
    q = "SELECT attendance.id, students.name, students.roll, students.class_section, attendance.timestamp, attendance.subject FROM attendance JOIN students ON attendance.student_id = students.id"
    conds = []
    params = []
    if date_from:
        conds.append("attendance.date >= ?")
        params.append(date_from)
    if date_to:
        conds.append("attendance.date <= ?")
        params.append(date_to)
    if class_section:
        conds.append("students.class_section = ?")
        params.append(class_section)
    if subject:
        conds.append("attendance.subject = ?")
        params.append(subject)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY attendance.timestamp DESC"
    with get_conn() as conn:
        c = conn.cursor()
        c.execute(q, params)
        rows = c.fetchall()
        return rows
