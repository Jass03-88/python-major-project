"""
Central configuration for the Biometric Attendance System.
Loads secrets and tunables from a local .env file (never commit that file).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Telegram ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- Admin auth (hashed, never plaintext) ---
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")
ADMIN_PASSWORD_SALT = os.getenv("ADMIN_PASSWORD_SALT", "")

# --- Lockout policy ---
LOCKOUT_THRESHOLD = int(os.getenv("LOCKOUT_THRESHOLD", 3))
LOCKOUT_WINDOW_SECONDS = int(os.getenv("LOCKOUT_WINDOW_SECONDS", 300))
LOCKOUT_DURATION_SECONDS = int(os.getenv("LOCKOUT_DURATION_SECONDS", 120))

# --- Recognition tuning ---
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.363))
FRAMES_TO_COLLECT = int(os.getenv("FRAMES_TO_COLLECT", 25))
LIVENESS_ENABLED = os.getenv("LIVENESS_ENABLED", "true").lower() == "true"

# --- Alerting ---
DENIAL_ALERT_COOLDOWN_SECONDS = int(os.getenv("DENIAL_ALERT_COOLDOWN_SECONDS", 60))

# --- Attendance ---
HALF_DAY_CUTOFF = os.getenv("HALF_DAY_CUTOFF", "16:30")
DB_PATH = os.getenv("ATTENDANCE_DB", "attendance.db")
