"""
Shared Telegram alerting helpers. Includes a cooldown specifically for
intruder-denial photos, so a burst of failed attempts (your logs show
several in under 5 minutes on 2026-07-08) sends one alert, not ten.
"""

import json
import os
import time
import requests

import config

ALERT_STATE_FILE = "alert_state.json"


def _configured() -> bool:
    return bool(config.TELEGRAM_BOT_TOKEN) and bool(config.TELEGRAM_CHAT_ID)


def _load_alert_state() -> dict:
    if not os.path.exists(ALERT_STATE_FILE):
        return {"last_denial_alert": 0}
    try:
        with open(ALERT_STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"last_denial_alert": 0}


def _save_alert_state(state: dict) -> None:
    with open(ALERT_STATE_FILE, "w") as f:
        json.dump(state, f)


def send_message(text: str) -> None:
    if not _configured():
        print("Telegram not configured — skipping message alert.")
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url, data={"chat_id": config.TELEGRAM_CHAT_ID, "text": text}, timeout=10
        )
        if resp.status_code != 200:
            print(f"Telegram message failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        print("Error sending Telegram message:", e)


def send_photo(photo_path: str, caption: str) -> None:
    if not _configured():
        print("Telegram not configured — skipping photo alert.")
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as photo:
            resp = requests.post(
                url,
                files={"photo": photo},
                data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": caption},
                timeout=10,
            )
        if resp.status_code == 200:
            print("Telegram photo alert sent.")
        else:
            print(f"Telegram photo alert failed ({resp.status_code}): {resp.text}")
    except Exception as e:
        print("Error sending Telegram photo alert:", e)


def send_denial_alert(photo_path: str, caption: str) -> None:
    """Rate-limited photo alert, used specifically for intruder denials."""
    state = _load_alert_state()
    now = time.time()
    if now - state.get("last_denial_alert", 0) < config.DENIAL_ALERT_COOLDOWN_SECONDS:
        print("Denial alert suppressed (cooldown active).")
        return
    send_photo(photo_path, caption)
    state["last_denial_alert"] = now
    _save_alert_state(state)
