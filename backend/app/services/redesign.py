"""
RedesignService: orchestrates segmentation mask retrieval + SD inpainting.
This is the main business logic for the /api/redesign endpoint.
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from app.ai.inpainting.sd_inpaint import run_sd_inpaint
from app.ai.loaders import model_store
from app.ai.styles_config import STYLE_CONFIG
from app.core.config import get_settings

settings = get_settings()

# Kitchen classes to restyle (must match labels in SegFormer output)
KITCHEN_CLASSES = ["floor", "cabinet", "backsplash", "countertop", "wall"]


def _pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def redesign_kitchen(
    original_image: Image.Image,
    masks: dict,
    style_name: str,
    target_classes: list[str] | None = None,
) -> dict:
    """
    Apply a style to the kitchen image using SD inpainting.
    Returns a dict with base64 before + after images.
    """
    if style_name not in STYLE_CONFIG:
        raise ValueError(
            f"Unknown style: {style_name}. Choose from: {list(STYLE_CONFIG.keys())}"
        )

    style_config = STYLE_CONFIG[style_name]
    pipeline = model_store.get_inpaint_pipeline(settings.inpaint_model)

    classes_to_restyle = target_classes or KITCHEN_CLASSES

    edited_image = run_sd_inpaint(
        pipeline=pipeline,
        original_image=original_image,
        masks=masks,
        style_config=style_config,
        target_classes=classes_to_restyle,
    )

    return {
        "style": style_name,
        "style_label": style_config["label"],
        "before_b64": _pil_to_b64(original_image),
        "after_b64": _pil_to_b64(edited_image),
    }
