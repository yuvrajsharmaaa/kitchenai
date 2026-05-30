"""
Stable Diffusion inpainting pipeline wrapper.
Uses runwayml/stable-diffusion-inpainting (or compatible checkpoint).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter
import torch


def _prepare_composite_mask(
    masks: dict[str, np.ndarray | Image.Image],
    target_classes: list[str],
) -> Image.Image:
    """
    Combine per-class binary masks into a single PIL mask image.
    White pixels = area to inpaint.
    """
    if not masks:
        raise ValueError("No masks provided")

    first = next(iter(masks.values()))
    first_arr = np.array(first)
    h, w = first_arr.shape[:2]
    composite = np.zeros((h, w), dtype=np.uint8)

    for cls in target_classes:
        if cls in masks:
            mask_arr = np.array(masks[cls])
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            composite = np.clip(
                composite + (mask_arr > 127).astype(np.uint8) * 255, 0, 255
            )

    # Dilate slightly for smoother blending
    pil_mask = Image.fromarray(composite.astype(np.uint8), mode="L")
    pil_mask = pil_mask.filter(ImageFilter.MaxFilter(5))
    pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=2))
    return pil_mask


def run_sd_inpaint(
    pipeline,
    original_image: Image.Image,
    masks: dict[str, np.ndarray | Image.Image],
    style_config: dict,
    target_classes: list[str] | None = None,
    seed: int = 42,
) -> Image.Image:
    """
    Run SD inpainting and return the edited image.
    The edited pixels are composited back onto the original
    outside the mask boundary for clean edges.
    """
    if target_classes is None:
        target_classes = ["floor", "cabinet", "backsplash", "countertop"]

    composite_mask = _prepare_composite_mask(masks, target_classes)

    # Resize to SD-compatible size (must be multiples of 8)
    orig_w, orig_h = original_image.size
    max_dim = 768 if torch.cuda.is_available() else 512
    sd_w = min(((orig_w // 8) * 8), max_dim)
    sd_h = min(((orig_h // 8) * 8), max_dim)

    image_resized = original_image.resize((sd_w, sd_h), Image.LANCZOS).convert("RGB")
    mask_resized = composite_mask.resize((sd_w, sd_h), Image.NEAREST)

    generator = torch.Generator().manual_seed(seed)

    num_steps = style_config.get("num_inference_steps", 30)
    if not torch.cuda.is_available():
        num_steps = min(num_steps, 20)

    result = pipeline(
        prompt=style_config["prompt"],
        negative_prompt=style_config["negative_prompt"],
        image=image_resized,
        mask_image=mask_resized,
        height=sd_h,
        width=sd_w,
        strength=style_config.get("strength", 0.80),
        guidance_scale=style_config.get("guidance_scale", 8.5),
        num_inference_steps=num_steps,
        generator=generator,
    ).images[0]

    result_orig_size = result.resize((orig_w, orig_h), Image.LANCZOS)
    composite_mask_orig = composite_mask.resize((orig_w, orig_h), Image.NEAREST)

    final = Image.composite(
        result_orig_size, original_image.convert("RGB"), composite_mask_orig
    )
    return final
