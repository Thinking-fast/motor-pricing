"""Load configuration from config.yaml and secrets from .env.

Import `load_config()` anywhere you need settings. Keeping all configuration in
one place (and out of the code) is a basic but important engineering habit.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Return the parsed config dict, with .env loaded into the environment."""
    load_dotenv(PROJECT_ROOT / ".env")  # makes OPENAI_API_KEY available if present
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg
