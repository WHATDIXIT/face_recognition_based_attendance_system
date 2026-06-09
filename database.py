"""
database.py
===========
SQLite database management for the Smart Attendance System.
Handles all DB operations: creation, CRUD for students and attendance.
"""

import sqlite3
import os
import hashlib
import pandas as pd
from datetime import datetime, date


# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_DIR     = os.path.join(BASE_DIR, "database")
DB_PATH    = os.path.join(DB_DIR, "attendance.db")


# ════════════════════════════════════════════════════════════════════════════
#  Core helpers
# ════════════════════════════════════════════════════════════════════════════

def get_connection() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection with foreign-key support."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row          # dict-like row access
    return conn


def _hash_password(password: str) -> str:
    """SHA-256 hash for admin passwords."""
    return hashlib.sha256(password.encode()).hexdigest()


# ════════════════════════════════════════════════════════════════════════════
#  Database initialisation
# ════════════════════════════════════════════════════════════════════════════

def init_database() -> None:
    """Create all tables and seed the default admin account."""
    conn = get_connection()
    cur  = conn.cursor()

    # ── Students ────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            roll_no     TEXT    NOT NULL UNIQUE,
            branch      TEXT    NOT NULL,
            year        TEXT    NOT NULL,
            registered_at TEXT  DEFAULT (datetime('now','localtime'))
        )
    """)

    # ── Attendance ──────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT    NOT NULL,
            name    TEXT    NOT NULL,
            date    TEXT    NOT NULL,
            time    TEXT    NOT NULL,
            status  TEXT    NOT NULL DEFAULT 'Present',
            UNIQUE(roll_no, date)
        )
    """)

    # ── Admin ───────────────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    NOT NULL UNIQUE,
            password TEXT    NOT NULL
        )
    """)

    # Seed default admin (admin / admin123)
    cur.execute(
        "INSERT OR IGNORE INTO admins (username, password) VALUES (?, ?)",
        ("admin", _hash_password("admin123"))
    )

    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════════════════
#  Admin authentication
# ════════════════════════════════════════════════════════════════════════════

def verify_admin(username: str, password: str) -> bool:
    """Return True if credentials match a stored admin record."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute(
        "SELECT id FROM admins WHERE username=? AND password=?",
        (username, _hash_password(password))
    )
    row  = cur.fetchone()
    conn.close()
    return row is not None


# ════════════════════════════════════════════════════════════════════════════
#  Student CRUD
# ════════════════════════════════════════════════════════════════════════════

def add_student(name: str, roll_no: str, branch: str, year: str) -> dict:
    """
    Insert a new student. Returns {'success': bool, 'message': str}.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO students (name, roll_no, branch, year) VALUES (?,?,?,?)",
            (name.strip(), roll_no.strip().upper(), branch.strip(), year.strip())
        )
        conn.commit()
        return {"success": True, "message": f"Student '{name}' registered successfully."}
    except sqlite3.IntegrityError:
        return {"success": False, "message": f"Roll number '{roll_no}' already exists."}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        conn.close()


def get_all_students() -> pd.DataFrame:
    """Return all students as a DataFrame."""
    conn = get_connection()
    df   = pd.read_sql_query("SELECT * FROM students ORDER BY registered_at DESC", conn)
    conn.close()
    return df


def get_student_by_roll(roll_no: str) -> dict | None:
    """Return a single student dict or None."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM students WHERE roll_no=?", (roll_no.upper(),))
    row  = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_student(roll_no: str) -> dict:
    """Delete student and their attendance records."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("DELETE FROM attendance WHERE roll_no=?", (roll_no,))
        cur.execute("DELETE FROM students    WHERE roll_no=?", (roll_no,))
        conn.commit()
        return {"success": True, "message": f"Student {roll_no} deleted."}
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        conn.close()


def get_student_count() -> int:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM students")
    n    = cur.fetchone()[0]
    conn.close()
    return n


# ════════════════════════════════════════════════════════════════════════════
#  Attendance CRUD
# ════════════════════════════════════════════════════════════════════════════

def mark_attendance(roll_no: str, name: str) -> dict:
    """
    Mark a student present for today.
    Duplicate entries (same roll_no + date) are silently ignored.
    """
    now    = datetime.now()
    today  = now.strftime("%Y-%m-%d")
    t_now  = now.strftime("%H:%M:%S")

    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute(
            "INSERT OR IGNORE INTO attendance (roll_no, name, date, time, status) "
            "VALUES (?,?,?,?,'Present')",
            (roll_no, name, today, t_now)
        )
        conn.commit()
        affected = cur.rowcount
        msg = (
            f"Attendance marked for {name} ({roll_no})"
            if affected else
            f"{name} already marked present today."
        )
        return {"success": True, "message": msg, "new_entry": bool(affected)}
    except Exception as e:
        return {"success": False, "message": str(e), "new_entry": False}
    finally:
        conn.close()


def get_all_attendance() -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql_query(
        "SELECT * FROM attendance ORDER BY date DESC, time DESC", conn
    )
    conn.close()
    return df


def get_attendance_by_date(target_date: str) -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql_query(
        "SELECT * FROM attendance WHERE date=? ORDER BY time",
        conn, params=(target_date,)
    )
    conn.close()
    return df


def get_attendance_by_roll(roll_no: str) -> pd.DataFrame:
    conn = get_connection()
    df   = pd.read_sql_query(
        "SELECT * FROM attendance WHERE roll_no=? ORDER BY date DESC",
        conn, params=(roll_no,)
    )
    conn.close()
    return df


def get_attendance_count() -> int:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM attendance")
    n    = cur.fetchone()[0]
    conn.close()
    return n


def get_today_attendance_count() -> int:
    today = date.today().strftime("%Y-%m-%d")
    conn  = get_connection()
    cur   = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,))
    n     = cur.fetchone()[0]
    conn.close()
    return n


def get_monthly_attendance(year: int, month: int) -> pd.DataFrame:
    prefix = f"{year}-{month:02d}"
    conn   = get_connection()
    df     = pd.read_sql_query(
        "SELECT * FROM attendance WHERE date LIKE ? ORDER BY date, time",
        conn, params=(f"{prefix}%",)
    )
    conn.close()
    return df


def get_attendance_stats() -> pd.DataFrame:
    """Per-student attendance percentage (vs. distinct days in the DB)."""
    conn  = get_connection()
    total_days_q = "SELECT COUNT(DISTINCT date) FROM attendance"
    cur   = conn.cursor()
    cur.execute(total_days_q)
    total_days = cur.fetchone()[0] or 1

    df = pd.read_sql_query("""
        SELECT roll_no, name,
               COUNT(*) AS days_present,
               ROUND(COUNT(*) * 100.0 / ?, 1) AS attendance_pct
        FROM   attendance
        GROUP  BY roll_no, name
        ORDER  BY attendance_pct DESC
    """, conn, params=(total_days,))
    conn.close()
    return df
