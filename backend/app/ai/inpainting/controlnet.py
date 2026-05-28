from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from app.ai.inpainting.config import DEFAULT_NEGATIVE_PROMPT, InpaintSettings
from app.ai.loaders import ModelStore


@dataclass
class InpaintResult:
    image: Image.Image


class ControlNetInpaint:
    def __init__(
        self,
        inpaint_model: str,
        controlnet_model: str,
        device: str,
        store: ModelStore,
    ) -> None:
        self.inpaint_model = inpaint_model
        self.controlnet_model = controlnet_model
        self.device = device
        self.store = store

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.5,
        controlnet_conditioning_scale: float = 0.7,
        settings: Optional[InpaintSettings] = None,
    ) -> InpaintResult:
        # Assemble settings so callers can override only what they need.
        if settings is None:
            settings = InpaintSettings(
                prompt=prompt,
                negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )

        bundle = self.store.get_inpaint(
            self.inpaint_model, self.controlnet_model, self.device
        )

        prepared_image, prepared_mask = prepare_inpaint_inputs(image, mask)
        prepared_mask = preprocess_mask(
            prepared_mask, settings.mask_blur, settings.mask_dilate
        )
        control_image = create_canny_control(
            prepared_image, settings.canny_low, settings.canny_high
        )

        generator = None
        if settings.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(settings.seed)

        result = bundle.pipeline(
            prompt=settings.prompt,
            negative_prompt=settings.negative_prompt,
            image=prepared_image,
            mask_image=prepared_mask,
            control_image=control_image,
            num_inference_steps=settings.num_inference_steps,
            guidance_scale=settings.guidance_scale,
            controlnet_conditioning_scale=settings.controlnet_conditioning_scale,
            generator=generator,
        )
        return InpaintResult(image=result.images[0])


def prepare_inpaint_inputs(
    image: Image.Image, mask: Image.Image
) -> Tuple[Image.Image, Image.Image]:
    # Resize to a multiple of 8 for Stable Diffusion compatibility.
    width, height = image.size
    target_width = width - (width % 8)
    target_height = height - (height % 8)
    if target_width <= 0 or target_height <= 0:
        target_width = max(8, width)
        target_height = max(8, height)
    if (target_width, target_height) != image.size:
        image = image.resize((target_width, target_height), Image.BICUBIC)
        mask = mask.resize((target_width, target_height), Image.NEAREST)
    return image.convert("RGB"), mask.convert("L")


def preprocess_mask(mask: Image.Image, blur_size: int, dilate_size: int) -> Image.Image:
    # Slightly blur and dilate mask edges for smoother blends.
    np_mask = np.array(mask)
    if blur_size > 0:
        if blur_size % 2 == 0:
            blur_size += 1
        np_mask = cv2.GaussianBlur(np_mask, (blur_size, blur_size), 0)
    if dilate_size > 0:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        np_mask = cv2.dilate(np_mask, kernel, iterations=1)
    return Image.fromarray(np_mask, mode="L")


def create_canny_control(
    image: Image.Image, low_threshold: int, high_threshold: int
) -> Image.Image:
    # Canny edges act as a geometry guide to preserve perspective and layout.
    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)
