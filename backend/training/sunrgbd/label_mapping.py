from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional


IGNORE_LABEL = 255

CLASS_ID_MAP: Dict[str, int] = {
    "background": 0,
    "floor": 1,
    "wall": 2,
    "cabinet": 3,
    "countertop": 4,
    "furniture": 5,
    "appliances": 6,
}

ID_TO_LABEL = {class_id: name for name, class_id in CLASS_ID_MAP.items()}

CLASS_COLORS: Dict[str, tuple[int, int, int]] = {
    "background": (0, 0, 0),
    "floor": (128, 64, 128),
    "wall": (70, 130, 180),
    "cabinet": (180, 120, 60),
    "countertop": (220, 180, 80),
    "furniture": (120, 170, 90),
    "appliances": (210, 80, 90),
}

TARGET_ALIASES: Dict[str, Iterable[str]] = {
    "floor": [
        "floor",
        "floors",
        "ground",
    ],
    "wall": [
        "wall",
        "walls",
    ],
    "cabinet": [
        "cabinet",
        "cabinetry",
        "cupboard",
        "cuboard",
        "closet",
        "locker",
        "drawer",
        "drawers",
    ],
    "countertop": [
        "counter",
        "countertop",
        "counter top",
        "worktop",
        "table counter",
    ],
    "furniture": [
        "chair",
        "stool",
        "table",
        "desk",
        "sofa",
        "couch",
        "bed",
        "bench",
        "shelf",
        "shelves",
        "bookshelf",
        "bookcase",
        "dresser",
        "nightstand",
        "stand",
        "ottoman",
        "wardrobe",
        "furniture",
    ],
    "appliances": [
        "appliance",
        "appliances",
        "refrigerator",
        "fridge",
        "freezer",
        "oven",
        "microwave",
        "stove",
        "range",
        "cooktop",
        "dishwasher",
        "washer",
        "dryer",
        "toaster",
        "kettle",
        "sink",
        "garbage disposal",
    ],
}


def normalize_label(name: object) -> str:
    """Return a simple lowercase label string for loose SUN RGB-D names."""
    if not isinstance(name, str):
        return ""
    cleaned = name.replace("_", " ").replace("-", " ").strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def label_to_target_class(label_name: object) -> Optional[str]:
    """Map one SUN RGB-D object name to a target class name."""
    normalized = normalize_label(label_name)
    if not normalized:
        return None

    for target_class, aliases in TARGET_ALIASES.items():
        for alias in aliases:
            alias = normalize_label(alias)
            if normalized == alias or alias in normalized:
                return target_class
    return None


def label_to_class_id(label_name: object) -> int:
    """Map one SUN RGB-D object name to a numeric class ID."""
    target_class = label_to_target_class(label_name)
    if target_class is None:
        return CLASS_ID_MAP["background"]
    return CLASS_ID_MAP[target_class]


def export_label_metadata(output_dir: Path) -> None:
    """Write class maps used by conversion, visualization, and training."""
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map_path = output_dir / "label_map.json"
    dataset_info_path = output_dir / "dataset_info.json"

    with label_map_path.open("w", encoding="utf-8") as handle:
        json.dump(CLASS_ID_MAP, handle, indent=2)

    dataset_info = {
        "labels": CLASS_ID_MAP,
        "id2label": {str(class_id): label for class_id, label in ID_TO_LABEL.items()},
        "label2id": CLASS_ID_MAP,
        "ignore_label": IGNORE_LABEL,
        "palette": {
            label: list(color) for label, color in CLASS_COLORS.items()
        },
    }
    with dataset_info_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_info, handle, indent=2)
