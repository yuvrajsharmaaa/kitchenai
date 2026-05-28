from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from diffusers.models.controlnet import ControlNetModel
    from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
        StableDiffusionControlNetInpaintPipeline,
    )
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
else:
    AutoImageProcessor = Any
    SegformerForSemanticSegmentation = Any
    ControlNetModel = Any
    StableDiffusionControlNetInpaintPipeline = Any


@dataclass
class SegmentationBundle:
    processor: Any
    model: Any


@dataclass
class InpaintBundle:
    pipeline: Any


class ModelStore:
    def __init__(self) -> None:
        self._segmentation: Optional[SegmentationBundle] = None
        self._inpaint: Optional[InpaintBundle] = None

    def get_segmentation(self, model_name: str, device: str) -> SegmentationBundle:
        if self._segmentation is not None:
            return self._segmentation
        from transformers import AutoImageProcessor as _AutoImageProcessor
        from transformers import SegformerForSemanticSegmentation as _SegformerForSemanticSegmentation

        processor = _AutoImageProcessor.from_pretrained(model_name)
        model = _SegformerForSemanticSegmentation.from_pretrained(model_name)
        model_any = cast(Any, model)
        model_any.to(device)
        model_any.eval()
        self._segmentation = SegmentationBundle(processor=processor, model=model_any)
        return self._segmentation

    def get_inpaint(
        self, inpaint_model: str, controlnet_model: str, device: str
    ) -> InpaintBundle:
        if self._inpaint is not None:
            return self._inpaint
        from diffusers.models.controlnet import ControlNetModel as _ControlNetModel
        from diffusers.pipelines.controlnet.pipeline_controlnet_inpaint import (
            StableDiffusionControlNetInpaintPipeline as _StableDiffusionControlNetInpaintPipeline,
        )

        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        controlnet = _ControlNetModel.from_pretrained(
            controlnet_model, torch_dtype=torch_dtype
        )
        pipe = _StableDiffusionControlNetInpaintPipeline.from_pretrained(
            inpaint_model,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
        )
        pipe.to(device)
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        self._inpaint = InpaintBundle(pipeline=pipe)
        return self._inpaint


model_store = ModelStore()
