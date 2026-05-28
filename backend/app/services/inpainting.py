from typing import Optional

import torch
from PIL import Image

from app.ai.inpainting.controlnet import ControlNetInpaint, InpaintResult
from app.ai.loaders import model_store


class InpaintingService:
    def __init__(self, inpaint_model: str, controlnet_model: str) -> None:
        self.inpaint_model = inpaint_model
        self.controlnet_model = controlnet_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._engine = ControlNetInpaint(
            inpaint_model=self.inpaint_model,
            controlnet_model=self.controlnet_model,
            device=self.device,
            store=model_store,
        )

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
        return self._engine.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
