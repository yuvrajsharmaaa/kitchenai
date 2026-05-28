from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
)

LABELS = ["background", "floor", "wall", "cabinet", "countertop", "furniture", "appliances"]


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        processor: SegformerImageProcessor,
        split_csv: Path | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.images: List[Path] = []
        self.masks: Dict[str, Path] = {}
        self.processor = processor

        if split_csv is not None and data_dir is not None and split_csv.exists():
            with split_csv.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    image_path = data_dir / row["image"]
                    mask_path = data_dir / row["mask"]
                    self.images.append(image_path)
                    self.masks[image_path.stem] = mask_path
        else:
            self.images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
            self.masks = {path.stem: path for path in masks_dir.glob("*.png")}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_path = self.images[index]
        mask_path = self.masks.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {image_path.name}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        encoded = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
        encoded = {key: value.squeeze(0) for key, value in encoded.items()}
        return encoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer on SUN RGB-D.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/segformer_sunrgbd"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


def load_labels(data_dir: Path) -> List[str]:
    dataset_info = data_dir / "metadata" / "dataset_info.json"
    if not dataset_info.exists():
        return LABELS

    with dataset_info.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    id2label = data.get("id2label", {})
    if not id2label:
        return LABELS
    return [id2label[str(index)] for index in range(len(id2label))]


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)
    preds = logits.argmax(dim=1)
    ious = []
    for label_id in range(len(LABELS)):
        pred_mask = preds == label_id
        label_mask = labels == label_id
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return {"mean_iou": mean_iou}


def main() -> None:
    global LABELS

    args = parse_args()
    labels = load_labels(args.data_dir)
    LABELS = labels

    train_images = args.data_dir / "train" / "images"
    train_masks = args.data_dir / "train" / "masks"
    val_images = args.data_dir / "val" / "images"
    val_masks = args.data_dir / "val" / "masks"
    train_csv = args.data_dir / "metadata" / "train.csv"
    val_csv = args.data_dir / "metadata" / "val.csv"

    processor = SegformerImageProcessor(size=args.image_size)

    train_dataset = SegmentationDataset(
        train_images,
        train_masks,
        processor,
        split_csv=train_csv,
        data_dir=args.data_dir,
    )
    val_dataset = SegmentationDataset(
        val_images,
        val_masks,
        processor,
        split_csv=val_csv,
        data_dir=args.data_dir,
    )

    id2label = {idx: name for idx, name in enumerate(LABELS)}
    label2id = {name: idx for idx, name in enumerate(LABELS)}

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
