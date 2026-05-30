from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

from app.ai.loaders import model_store


@dataclass
class InpaintResult:
    image: Image.Image


class InpaintingService:
    def __init__(self, inpaint_model: str) -> None:
        self.inpaint_model = inpaint_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline = model_store.get_inpaint_pipeline(
            self.inpaint_model
        ).pipeline

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.5,
        controlnet_conditioning_scale: float = 0.7,
    ) -> InpaintResult:
        _ = controlnet_conditioning_scale

        base_image = image.convert("RGB")
        mask_l = mask.convert("L")

        orig_w, orig_h = base_image.size
        target_w = orig_w - (orig_w % 8)
        target_h = orig_h - (orig_h % 8)
        if target_w <= 0 or target_h <= 0:
            target_w = max(8, orig_w)
            target_h = max(8, orig_h)

        if (target_w, target_h) != base_image.size:
            base_image = base_image.resize((target_w, target_h), Image.LANCZOS)
            mask_l = mask_l.resize((target_w, target_h), Image.NEAREST)

        result = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            mask_image=mask_l,
            height=base_image.height,
            width=base_image.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        result = result.resize((orig_w, orig_h), Image.LANCZOS)
        mask_full = mask_l.resize((orig_w, orig_h), Image.NEAREST)
        final = Image.composite(result, image.convert("RGB"), mask_full)

        return InpaintResult(image=final)
