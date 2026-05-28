from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SUN RGB-D style data for SegFormer training."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path(__file__).parent / "configs" / "label_map.json",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["id", "rgb"],
        default="id",
        help="Use 'id' for single-channel class IDs or 'rgb' for color-coded masks.",
    )
    return parser.parse_args()


def load_label_map(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_rgb_palette(label_map: Dict[str, int]) -> Dict[Tuple[int, int, int], int]:
    # You can customize this palette to match your dataset's RGB label colors.
    # Defaults are placeholders and should be updated to real SUN RGB-D colors.
    return {
        (0, 0, 0): label_map["background"],
        (128, 64, 128): label_map["floor"],
        (70, 130, 180): label_map["wall"],
        (180, 120, 60): label_map["cabinet"],
        (220, 180, 80): label_map["countertop"],
        (120, 170, 90): label_map["furniture"],
        (210, 80, 90): label_map["appliances"],
    }


def remap_mask(mask: Image.Image, palette: Dict[Tuple[int, int, int], int]) -> Image.Image:
    np_mask = np.array(mask)
    if np_mask.ndim == 2:
        return Image.fromarray(np_mask.astype(np.uint8), mode="L")

    output = np.zeros((np_mask.shape[0], np_mask.shape[1]), dtype=np.uint8)
    for color, target_id in palette.items():
        matches = np.all(np_mask == np.array(color, dtype=np.uint8), axis=-1)
        output[matches] = target_id
    return Image.fromarray(output, mode="L")


def pair_files(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    images = sorted(images_dir.rglob("*.jpg")) + sorted(images_dir.rglob("*.png"))
    pairs: List[Tuple[Path, Path]] = []
    for image_path in images:
        mask_path = masks_dir / f"{image_path.stem}.png"
        if mask_path.exists():
            pairs.append((image_path, mask_path))
    return pairs


def split_pairs(pairs: List[Tuple[Path, Path]], val_split: float, seed: int) -> Tuple[List, List]:
    rng = random.Random(seed)
    rng.shuffle(pairs)
    val_count = max(1, int(len(pairs) * val_split))
    return pairs[val_count:], pairs[:val_count]


def ensure_dirs(base: Path) -> Dict[str, Path]:
    train_images = base / "train" / "images"
    train_masks = base / "train" / "masks"
    val_images = base / "val" / "images"
    val_masks = base / "val" / "masks"
    for path in [train_images, train_masks, val_images, val_masks]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "train_images": train_images,
        "train_masks": train_masks,
        "val_images": val_images,
        "val_masks": val_masks,
    }


def copy_pairs(
    pairs: List[Tuple[Path, Path]],
    out_images: Path,
    out_masks: Path,
    palette: Dict[Tuple[int, int, int], int],
    mask_mode: str,
) -> None:
    for image_path, mask_path in pairs:
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if mask_mode == "rgb":
            mask = remap_mask(mask, palette)
        else:
            mask = mask.convert("L")
        image.save(out_images / f"{image_path.stem}.jpg", format="JPEG")
        mask.save(out_masks / f"{image_path.stem}.png", format="PNG")


def main() -> None:
    args = parse_args()
    label_map = load_label_map(args.label_map)
    palette = build_rgb_palette(label_map)

    pairs = pair_files(args.images_dir, args.masks_dir)
    if not pairs:
        raise SystemExit("No image/mask pairs found. Check your paths.")

    train_pairs, val_pairs = split_pairs(pairs, args.val_split, args.seed)
    output_dirs = ensure_dirs(args.output_dir)

    copy_pairs(train_pairs, output_dirs["train_images"], output_dirs["train_masks"], palette, args.mask_mode)
    copy_pairs(val_pairs, output_dirs["val_images"], output_dirs["val_masks"], palette, args.mask_mode)

    print(f"Prepared {len(train_pairs)} train and {len(val_pairs)} val samples.")


if __name__ == "__main__":
    main()
