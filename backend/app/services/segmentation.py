import torch
from PIL import Image

from app.ai.loaders import model_store
from app.ai.segmentation.segformer import SegFormerSegmenter, SegmentationArtifacts


class SegmentationService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._segmenter = SegFormerSegmenter(
            model_name=self.model_name,
            device=self.device,
            store=model_store,
        )

    def segment(self, image: Image.Image) -> SegmentationArtifacts:
        return self._segmenter.segment(image)
