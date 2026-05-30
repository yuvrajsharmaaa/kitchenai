import uuid

import torch
from PIL import Image

from app.utils.paths import get_outputs_dir

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

    def persist_session(
        self, image: Image.Image, artifacts: SegmentationArtifacts
    ) -> str:
        session_id = uuid.uuid4().hex[:8]
        session_dir = get_outputs_dir() / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        image.save(session_dir / "original.jpg", quality=95)

        for class_name, mask_data in artifacts.masks.items():
            mask_l = mask_data.convert("L")
            mask_l.save(session_dir / f"mask_{class_name}.png")

        return session_id
