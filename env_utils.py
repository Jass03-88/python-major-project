"""
Shared helper for safely reading/writing individual keys in the .env file.
Uses python-dotenv to handle complex formatting and quoting correctly.
"""

import os
import dotenv

ENV_PATH: str = ".env"


def _ensure_env_exists() -> None:
    if not os.path.exists(ENV_PATH):
        with open(ENV_PATH, "w") as f:
            f.write("")


def upsert_env_var(key: str, value: str) -> None:
    """Adds or updates a key-value pair in the .env file."""
    _ensure_env_exists()
    # set_key safely updates the file preserving other comments/spacing
    dotenv.set_key(ENV_PATH, key, str(value))


def read_env_var(key: str, default: str = "") -> str:
    """Reads a specific key from the .env file."""
    if not os.path.exists(ENV_PATH):
        return default
    val = dotenv.get_key(ENV_PATH, key)
    return val if val is not None else default
