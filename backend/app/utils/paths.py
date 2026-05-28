from __future__ import annotations

from pathlib import Path


def get_storage_root() -> Path:
    return Path(__file__).resolve().parents[1] / "storage"


def get_uploads_dir() -> Path:
    return get_storage_root() / "uploads"


def get_outputs_dir() -> Path:
    return get_storage_root() / "outputs"
