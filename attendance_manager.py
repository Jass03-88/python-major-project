"""
SQLite-backed attendance tracking. Replaces attendance.csv to avoid
write corruption if the app is ever triggered from more than one place
at once, and fixes the old silent no-op on a third scan: now a third
scan (after a completed check-in/check-out pair) starts a brand-new
session for the day instead of doing nothing.
"""

import sqlite3
from datetime import datetime

import config


def _connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            check_in TEXT NOT NULL,
            check_out TEXT,
            status TEXT
        )
    """)
    return conn


def _compute_status(hour_min):
    return "Half Day" if hour_min < config.HALF_DAY_CUTOFF else "Present"


def mark_attendance(name):
    """
    - No open session today       -> start a new session (check-in)
    - Open session (no check-out) -> close it (check-out)
    - Already completed today     -> start a NEW session (2nd/3rd/... entry)
    Returns a short human-readable description of what happened.
    """
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")
    hour_min = now.strftime("%H:%M")

    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, check_out FROM attendance WHERE name=? AND date=? ORDER BY id DESC LIMIT 1",
        (name, today),
    )
    row = cur.fetchone()

    if row is None:
        cur.execute(
            "INSERT INTO attendance (name, date, check_in, check_out, status) VALUES (?,?,?,?,?)",
            (name, today, current_time, None, _compute_status(hour_min)),
        )
        result = f"Checked in at {current_time}"
    elif row[1] is None:
        cur.execute(
            "UPDATE attendance SET check_out=?, status=? WHERE id=?",
            (current_time, _compute_status(hour_min), row[0]),
        )
        result = f"Checked out at {current_time}"
    else:
        cur.execute(
            "INSERT INTO attendance (name, date, check_in, check_out, status) VALUES (?,?,?,?,?)",
            (name, today, current_time, None, _compute_status(hour_min)),
        )
        result = f"New session started (check-in) at {current_time}"

    conn.commit()
    conn.close()
    return result


def get_today_summary():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT name, check_in, check_out, status FROM attendance WHERE date=? ORDER BY name, id",
        (today,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def export_to_csv(csv_path="attendance_export.csv"):
    """Optional: dump the whole table to a spreadsheet-friendly CSV."""
    import csv

    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT name, date, check_in, check_out, status FROM attendance ORDER BY date, name, id"
    )
    rows = cur.fetchall()
    conn.close()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Check-In", "Check-Out", "Status"])
        writer.writerows(rows)
    return csv_path

def get_daily_counts(days=7):
    """Returns a list of (date, distinct_people_count) for the last `days` days."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT date, COUNT(DISTINCT name) FROM attendance GROUP BY date ORDER BY date DESC LIMIT ?",
        (days,),
    )
    rows = cur.fetchall()
    conn.close()
    return list(reversed(rows))
