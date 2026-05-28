from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from app.utils.paths import get_outputs_dir, get_uploads_dir


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_upload(image: Image.Image, name_hint: str) -> Path:
    uploads_dir = get_uploads_dir()
    _ensure_dir(uploads_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = uploads_dir / f"{timestamp}_{name_hint}.png"
    image.save(file_path, format="PNG")
    return file_path


def save_output(image: Image.Image, name_hint: str) -> Path:
    outputs_dir = get_outputs_dir()
    _ensure_dir(outputs_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = outputs_dir / f"{timestamp}_{name_hint}.png"
    image.save(file_path, format="PNG")
    return file_path


def save_json(data: object, name_hint: str) -> Path:
    outputs_dir = get_outputs_dir()
    _ensure_dir(outputs_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = outputs_dir / f"{timestamp}_{name_hint}.json"
    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path


def to_relative(path: Path) -> str:
    root = get_uploads_dir().parent
    return str(path.relative_to(root))
