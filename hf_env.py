"""Load HuggingFace settings from a local ``.env`` file.

Reads ``HF_TOKEN`` and ``HF_CACHE`` from ``.env`` (see ``.env.example``) and exports the
standard HuggingFace environment variables (auth token + ``HF_HOME`` cache location).
Call this before importing ``transformers``/``huggingface_hub``, which read them at
import time.
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent / ".env"


def load_env(env_file: Path = ENV_FILE) -> None:
    if env_file.is_file():
        for raw_line in env_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    token = os.environ.get("HF_TOKEN")
    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    cache = os.environ.get("HF_CACHE")
    if cache:
        os.environ.setdefault("HF_HOME", cache)
