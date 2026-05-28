from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/validation split metadata for a converted SUN RGB-D dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_samples(samples_csv: Path) -> List[Dict[str, str]]:
    with samples_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def split_samples(
    samples: List[Dict[str, str]],
    val_split: float,
    seed: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not 0 < val_split < 1:
        raise ValueError("--val-split must be between 0 and 1.")

    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_split)))
    return shuffled[val_count:], shuffled[:val_count]


def write_split_files(
    dataset_dir: Path,
    split_name: str,
    rows: List[Dict[str, str]],
) -> None:
    metadata_dir = dataset_dir / "metadata"
    txt_path = metadata_dir / f"{split_name}.txt"
    csv_path = metadata_dir / f"{split_name}.csv"
    jsonl_path = metadata_dir / f"{split_name}.jsonl"

    with txt_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['id']}\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "image": row["image"],
                        "mask": row["mask"],
                    }
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    samples_csv = args.dataset_dir / "metadata" / "samples.csv"
    if not samples_csv.exists():
        raise SystemExit(
            f"Missing {samples_csv}. Run convert_sunrgbd.py before creating splits."
        )

    samples = read_samples(samples_csv)
    if len(samples) < 2:
        raise SystemExit("Need at least two samples to create a train/validation split.")

    train_rows, val_rows = split_samples(samples, args.val_split, args.seed)
    write_split_files(args.dataset_dir, "train", train_rows)
    write_split_files(args.dataset_dir, "val", val_rows)
    print(f"Wrote {len(train_rows)} train and {len(val_rows)} validation samples.")


if __name__ == "__main__":
    main()
