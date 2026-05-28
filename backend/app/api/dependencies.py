from app.core.config import get_settings
from app.services.inpainting import InpaintingService
from app.services.segmentation import SegmentationService


def get_segmentation_service() -> SegmentationService:
    settings = get_settings()
    return SegmentationService(settings.segmentation_model)


def get_inpainting_service() -> InpaintingService:
    settings = get_settings()
    return InpaintingService(settings.inpaint_model, settings.controlnet_model)
