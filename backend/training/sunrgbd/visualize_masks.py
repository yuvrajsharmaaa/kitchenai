from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

from label_mapping import CLASS_COLORS, ID_TO_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize converted SUN RGB-D masks as color overlays."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <dataset-dir>/metadata/visualizations.",
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_samples(dataset_dir: Path) -> Dict[str, Dict[str, str]]:
    samples_csv = dataset_dir / "metadata" / "samples.csv"
    with samples_csv.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, label in ID_TO_LABEL.items():
        color_mask[mask == class_id] = CLASS_COLORS[label]
    return color_mask


def save_visualization(
    dataset_dir: Path,
    row: Dict[str, str],
    output_dir: Path,
    alpha: float,
) -> Path:
    image_path = dataset_dir / row["image"]
    mask_path = dataset_dir / row["mask"]

    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    color_mask = colorize_mask(mask)
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    output = np.concatenate([image, color_mask, overlay], axis=1)
    output_path = output_dir / f"{row['id']}_visualization.jpg"
    Image.fromarray(output).save(output_path, quality=95)
    return output_path


def choose_rows(
    samples: Dict[str, Dict[str, str]],
    sample_id: Optional[str],
    count: int,
    seed: int,
) -> list[Dict[str, str]]:
    if sample_id:
        if sample_id not in samples:
            raise KeyError(f"Unknown sample id: {sample_id}")
        return [samples[sample_id]]

    rows = list(samples.values())
    random.Random(seed).shuffle(rows)
    return rows[:count]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.dataset_dir / "metadata" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = read_samples(args.dataset_dir)
    rows = choose_rows(samples, args.sample_id, args.count, args.seed)

    for row in rows:
        output_path = save_visualization(args.dataset_dir, row, output_dir, args.alpha)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
