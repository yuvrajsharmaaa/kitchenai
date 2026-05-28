from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_NEGATIVE_PROMPT = (
    "distorted geometry, warped cabinets, broken perspective, "
    "low quality, blurry, noisy, artifacts"
)


@dataclass
class InpaintSettings:
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    num_inference_steps: int = 30
    guidance_scale: float = 6.5
    controlnet_conditioning_scale: float = 0.8
    canny_low: int = 100
    canny_high: int = 200
    mask_blur: int = 5
    mask_dilate: int = 3
    seed: Optional[int] = None
