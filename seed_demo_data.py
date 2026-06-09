"""
seed_demo_data.py
=================
Populates the database with realistic demo data for testing / screenshots.
Run once:  python seed_demo_data.py
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date, timedelta
from database import init_database, add_student, mark_attendance, get_connection

STUDENTS = [
    ("Arjun Sharma",      "CS2021001", "Computer Science", "4th Year"),
    ("Priya Verma",       "CS2021002", "Computer Science", "4th Year"),
    ("Rahul Gupta",       "AI2021001", "AI & ML",          "4th Year"),
    ("Anjali Singh",      "AI2021002", "AI & ML",          "4th Year"),
    ("Vikram Patel",      "IT2021001", "Information Technology", "4th Year"),
    ("Neha Joshi",        "IT2021002", "Information Technology", "4th Year"),
    ("Aditya Kumar",      "EC2021001", "Electronics & Communication", "4th Year"),
    ("Sneha Reddy",       "ME2021001", "Mechanical Engineering", "4th Year"),
]

# Attendance rates: index → probability of being present on any given day
RATES = [0.95, 0.90, 0.88, 0.75, 0.85, 0.70, 0.92, 0.60]


def seed():
    init_database()

    print("Seeding students…")
    for student in STUDENTS:
        res = add_student(*student)
        print(f"  {res['message']}")

    print("\nSeeding attendance (last 30 days)…")
    today = date.today()
    conn  = get_connection()
    cur   = conn.cursor()

    for i, (name, roll, _, _) in enumerate(STUDENTS):
        rate = RATES[i]
        for days_ago in range(29, -1, -1):
            day = today - timedelta(days=days_ago)
            # Skip weekends
            if day.weekday() >= 5:
                continue
            if random.random() < rate:
                t    = f"{random.randint(8,9):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}"
                date_str = day.strftime("%Y-%m-%d")
                cur.execute(
                    "INSERT OR IGNORE INTO attendance (roll_no, name, date, time, status) "
                    "VALUES (?,?,?,?,'Present')",
                    (roll, name, date_str, t)
                )

    conn.commit()
    conn.close()
    print("Done! Demo data seeded successfully.")


if __name__ == "__main__":
    seed()
