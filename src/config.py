"""
config.py — Loads config/config.yaml and exposes a single CFG object.

Usage in any module:
    from src.config import CFG
    model_name = CFG["embedding"]["model"]
"""

import yaml
import os
from pathlib import Path

# Always resolve relative to this file's location
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


CFG = load_config()