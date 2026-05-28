from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from app.utils.paths import get_outputs_dir

matplotlib.use("Agg")


def save_segmentation_preview(
    image: Image.Image,
    overlay: Image.Image,
    segmentation_map: Image.Image,
    name_hint: str,
) -> Path:
    outputs_dir = get_outputs_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    preview_path = outputs_dir / f"{timestamp}_{name_hint}.png"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(segmentation_map)
    axes[1].set_title("Segmentation map")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(preview_path, dpi=160)
    plt.close(fig)
    return preview_path
