from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
else:
    AutoImageProcessor = Any
    SegformerForSemanticSegmentation = Any


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
        self._sd_inpaint: Optional[InpaintBundle] = None

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

    def get_inpaint_pipeline(self, inpaint_model: str) -> InpaintBundle:
        if self._sd_inpaint is not None:
            return self._sd_inpaint
        from diffusers import StableDiffusionInpaintPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpaint_model,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        pipe.to(device)
        pipe.enable_attention_slicing()

        self._sd_inpaint = InpaintBundle(pipeline=pipe)
        return self._sd_inpaint


model_store = ModelStore()
