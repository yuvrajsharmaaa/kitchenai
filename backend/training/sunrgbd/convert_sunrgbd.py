from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from PIL import Image

from label_mapping import CLASS_ID_MAP, export_label_metadata, label_to_class_id


ANNOTATION_DIRS = ("annotation2Dfinal", "annotation2D3D", "annotation")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class SceneRecord:
    scene_dir: Path
    annotation_path: Path
    image_path: Path
    output_stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert original SUN RGB-D polygon annotations into PNG class masks."
    )
    parser.add_argument(
        "--sunrgbd-root",
        type=Path,
        default=Path(r"C:\Users\Asus\Desktop\kitchyen\SUNRGB\SUNRGBD\SUNRGBD"),
        help="Path to the extracted SUNRGBD folder that contains kv1, kv2, realsense, xtion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Output folder. The script creates images, masks, and metadata inside it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional small limit for quick testing.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip scenes where none of the target classes are present.",
    )
    return parser.parse_args()


def find_annotation(scene_dir: Path) -> Optional[Path]:
    for annotation_dir in ANNOTATION_DIRS:
        candidate = scene_dir / annotation_dir / "index.json"
        if candidate.exists():
            return candidate
    return None


def find_image(scene_dir: Path, annotation: Dict) -> Optional[Path]:
    file_list = annotation.get("fileList") or []
    for file_name in file_list:
        candidate = scene_dir / "image" / Path(str(file_name)).name
        if candidate.exists():
            return candidate

    for folder_name in ("image", "fullres"):
        image_dir = scene_dir / folder_name
        if not image_dir.exists():
            continue
        for extension in IMAGE_EXTENSIONS:
            matches = sorted(image_dir.glob(f"*{extension}"))
            if matches:
                return matches[0]
    return None


def make_output_stem(scene_dir: Path, sunrgbd_root: Path) -> str:
    relative = scene_dir.relative_to(sunrgbd_root)
    parts = [part for part in relative.parts if part]
    readable = "__".join(parts[-4:])
    digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{readable}__{digest}"


def iter_scene_records(sunrgbd_root: Path) -> Iterable[SceneRecord]:
    for annotation_path in sorted(sunrgbd_root.rglob("index.json")):
        if annotation_path.parent.name not in ANNOTATION_DIRS:
            continue
        scene_dir = annotation_path.parent.parent
        preferred_annotation = find_annotation(scene_dir)
        if preferred_annotation != annotation_path:
            continue

        with annotation_path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        image_path = find_image(scene_dir, annotation)
        if image_path is None:
            continue

        yield SceneRecord(
            scene_dir=scene_dir,
            annotation_path=annotation_path,
            image_path=image_path,
            output_stem=make_output_stem(scene_dir, sunrgbd_root),
        )


def object_name_by_index(objects: List[object], object_index: int) -> str:
    if object_index < 0 or object_index >= len(objects):
        return ""
    value = objects[object_index]
    if isinstance(value, dict):
        return str(value.get("name", ""))
    return ""


def polygon_points(polygon: Dict, width: int, height: int) -> Optional[np.ndarray]:
    xs = polygon.get("x")
    ys = polygon.get("y")
    if not xs or not ys or len(xs) != len(ys):
        return None
    points = np.array(
        [[int(round(float(x))), int(round(float(y)))] for x, y in zip(xs, ys)],
        dtype=np.int32,
    )
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    if len(np.unique(points, axis=0)) < 3:
        return None
    return points.reshape((-1, 1, 2))


def rasterize_mask(annotation_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    objects = annotation.get("objects") or []

    for frame in annotation.get("frames", []):
        for polygon in frame.get("polygon", []):
            object_index = polygon.get("object")
            if object_index is None:
                continue
            label_name = object_name_by_index(objects, int(object_index))
            class_id = label_to_class_id(label_name)
            if class_id == CLASS_ID_MAP["background"]:
                continue
            points = polygon_points(polygon, width, height)
            if points is not None:
                cv2.fillPoly(mask, [points], int(class_id))
    return mask


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    dirs = {
        "images": output_dir / "images",
        "masks": output_dir / "masks",
        "metadata": output_dir / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_rgb_copy(source: Path, destination: Path) -> tuple[int, int]:
    image = Image.open(source).convert("RGB")
    image.save(destination, format="JPEG", quality=95)
    return image.size


def write_metadata(rows: List[Dict[str, object]], metadata_dir: Path) -> None:
    csv_path = metadata_dir / "samples.csv"
    jsonl_path = metadata_dir / "samples.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "image",
                "mask",
                "source_image",
                "source_annotation",
                "width",
                "height",
                "present_class_ids",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def convert_dataset(
    sunrgbd_root: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    skip_empty: bool = False,
) -> None:
    dirs = ensure_output_dirs(output_dir)
    export_label_metadata(dirs["metadata"])

    rows: List[Dict[str, object]] = []
    records = iter_scene_records(sunrgbd_root)

    for index, record in enumerate(records):
        if limit is not None and len(rows) >= limit:
            break

        output_image = dirs["images"] / f"{record.output_stem}.jpg"
        output_mask = dirs["masks"] / f"{record.output_stem}.png"

        width, height = save_rgb_copy(record.image_path, output_image)
        mask = rasterize_mask(record.annotation_path, (width, height))
        present_ids = sorted(int(value) for value in np.unique(mask) if value != 0)

        if skip_empty and not present_ids:
            output_image.unlink(missing_ok=True)
            continue

        Image.fromarray(mask, mode="L").save(output_mask, format="PNG")

        rows.append(
            {
                "id": record.output_stem,
                "image": f"images/{output_image.name}",
                "mask": f"masks/{output_mask.name}",
                "source_image": str(record.image_path),
                "source_annotation": str(record.annotation_path),
                "width": width,
                "height": height,
                "present_class_ids": " ".join(str(value) for value in present_ids),
            }
        )

        if len(rows) % 100 == 0:
            print(f"Converted {len(rows)} scenes...")

    write_metadata(rows, dirs["metadata"])
    print(f"Done. Converted {len(rows)} samples into {output_dir}.")


def main() -> None:
    args = parse_args()
    if not args.sunrgbd_root.exists():
        raise SystemExit(f"SUN RGB-D root does not exist: {args.sunrgbd_root}")
    convert_dataset(args.sunrgbd_root, args.output_dir, args.limit, args.skip_empty)


if __name__ == "__main__":
    main()
