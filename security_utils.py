"""
Password hashing + brute-force lockout tracking for the admin panel.
No plaintext passwords are ever stored — only a salted PBKDF2 hash.
"""

import hashlib
import json
import os
import time
import secrets as pysecrets

LOCKOUT_STATE_FILE = "lockout_state.json"


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100_000
    ).hex()


def generate_salt() -> str:
    return pysecrets.token_hex(16)


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    if not salt or not expected_hash:
        return False
    return hash_password(password, salt) == expected_hash


# ---------------- Lockout tracking ----------------


def _load_state() -> dict:
    if not os.path.exists(LOCKOUT_STATE_FILE):
        return {"failures": [], "locked_until": 0}
    try:
        with open(LOCKOUT_STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"failures": [], "locked_until": 0}


def _save_state(state: dict) -> None:
    with open(LOCKOUT_STATE_FILE, "w") as f:
        json.dump(state, f)


def is_locked_out() -> bool:
    state = _load_state()
    return time.time() < state.get("locked_until", 0)


def seconds_until_unlock() -> int:
    state = _load_state()
    remaining = state.get("locked_until", 0) - time.time()
    return max(0, int(remaining))


def record_failure(threshold: int, window_seconds: int, lockout_duration: int) -> bool:
    """Record a failed attempt. Returns True if this failure just
    triggered a brand-new lockout (useful for firing a one-time alert)."""
    state = _load_state()
    now = time.time()
    state["failures"] = [
        t for t in state.get("failures", []) if now - t < window_seconds
    ]
    state["failures"].append(now)
    triggered = False
    if len(state["failures"]) >= threshold:
        state["locked_until"] = now + lockout_duration
        state["failures"] = []
        triggered = True
    _save_state(state)
    return triggered


def reset_failures() -> None:
    state = _load_state()
    state["failures"] = []
    _save_state(state)
