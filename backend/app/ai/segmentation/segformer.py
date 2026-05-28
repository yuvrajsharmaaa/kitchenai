from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, cast

import numpy as np
import torch
from PIL import Image

from app.ai.loaders import ModelStore
from app.core.constants import LABEL_COLORS, TARGET_LABELS


@dataclass
class SegmentationArtifacts:
    masks: Dict[str, Image.Image]
    overlay: Image.Image
    segmentation_map: Image.Image
    pixel_counts: Dict[str, int]


class SegFormerSegmenter:
    def __init__(self, model_name: str, device: str, store: ModelStore) -> None:
        self.model_name = model_name
        self.device = device
        self.store = store

    def segment(self, image: Image.Image) -> SegmentationArtifacts:
        bundle = self.store.get_segmentation(self.model_name, self.device)

        processor = cast(Any, bundle.processor)
        model = cast(Any, bundle.model)

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        prediction = logits.argmax(dim=1)[0].cpu().numpy()

        label_ids = self._label_ids(model.config.id2label)
        masks: Dict[str, Image.Image] = {}
        pixel_counts: Dict[str, int] = {}
        for label in TARGET_LABELS:
            label_id = label_ids.get(label)
            if label_id is None:
                continue
            mask = (prediction == label_id).astype(np.uint8) * 255
            masks[label] = Image.fromarray(mask, mode="L")
            pixel_counts[label] = int(mask.sum() // 255)

        segmentation_map = self._build_segmentation_map(
            prediction, label_ids, image.size
        )
        overlay = self._build_overlay(image, masks)
        return SegmentationArtifacts(
            masks=masks,
            overlay=overlay,
            segmentation_map=segmentation_map,
            pixel_counts=pixel_counts,
        )

    def _label_ids(self, id2label: Dict[int, str]) -> Dict[str, int]:
        label_map: Dict[str, int] = {}
        for idx, name in id2label.items():
            label_map[str(name).lower()] = int(idx)
        return label_map

    def _build_overlay(
        self, image: Image.Image, masks: Dict[str, Image.Image]
    ) -> Image.Image:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        for label, mask in masks.items():
            color = LABEL_COLORS.get(label, (255, 255, 255))
            color_layer = Image.new("RGBA", image.size, (*color, 120))
            overlay = Image.composite(color_layer, overlay, mask)
        combined = Image.alpha_composite(image.convert("RGBA"), overlay)
        return combined.convert("RGB")

    def _build_segmentation_map(
        self,
        prediction: np.ndarray,
        label_ids: Dict[str, int],
        size: tuple[int, int],
    ) -> Image.Image:
        width, height = size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for label, label_id in label_ids.items():
            if label not in TARGET_LABELS:
                continue
            color = LABEL_COLORS.get(label, (255, 255, 255))
            mask = prediction == label_id
            canvas[mask] = np.array(color, dtype=np.uint8)
        return Image.fromarray(canvas, mode="RGB")
