from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies import get_inpainting_service, get_segmentation_service
from app.core.config import get_settings
from app.ai.inpainting.prompts import MATERIAL_PROMPTS
from app.schemas import (
    ImageResponse,
    PresetItem,
    PresetResponse,
    SegmentResponse,
    SegmentMetadata,
    UploadResponse,
)
from app.services.segmentation import SegmentationService
from app.utils.files import save_json, save_output, save_upload, to_relative
from app.utils.image_io import pil_to_base64_png, read_image_upload
from app.utils.visualization import save_segmentation_preview
from PIL import Image

if TYPE_CHECKING:
    from app.services.inpainting import InpaintingService

settings = get_settings()
router = APIRouter(prefix=settings.api_prefix)
logger = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/presets", response_model=PresetResponse)
async def list_presets() -> PresetResponse:
    items = [PresetItem(key=key, prompt=prompt) for key, prompt in MATERIAL_PROMPTS.items()]
    return PresetResponse(items=items)


@router.post("/upload", response_model=UploadResponse)
async def upload_image(image: UploadFile = File(...)) -> UploadResponse:
    try:
        # Read the uploaded file into a PIL image.
        pil_image = await read_image_upload(image)
        # Save the original upload for traceability.
        upload_path = save_upload(pil_image, "kitchen")
        return UploadResponse(
            width=pil_image.width,
            height=pil_image.height,
            upload_path=to_relative(upload_path),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Upload failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/segment", response_model=SegmentResponse)
async def segment(
    image: UploadFile = File(...),
    service: SegmentationService = Depends(get_segmentation_service),
) -> SegmentResponse:
    try:
        # Load and persist the upload before running AI.
        pil_image = await read_image_upload(image)
        upload_path = save_upload(pil_image, "kitchen")
        # Run segmentation once and reuse artifacts for all outputs.
        artifacts = service.segment(pil_image)
        overlay_path = save_output(artifacts.overlay, "segmentation_overlay")
        map_path = save_output(artifacts.segmentation_map, "segmentation_map")
        mask_paths = {
            label: save_output(mask, f"mask_{label}")
            for label, mask in artifacts.masks.items()
        }
        # Save a quick preview grid for debugging.
        preview_path = save_segmentation_preview(
            image=pil_image,
            overlay=artifacts.overlay,
            segmentation_map=artifacts.segmentation_map,
            name_hint="segmentation_preview",
        )

        mask_payload = {
            label: pil_to_base64_png(mask) for label, mask in artifacts.masks.items()
        }
        overlay = pil_to_base64_png(artifacts.overlay)
        segmentation_map = pil_to_base64_png(artifacts.segmentation_map)

        metadata = SegmentMetadata(
            model=settings.segmentation_model,
            device=service.device,
            labels=list(artifacts.masks.keys()),
            width=pil_image.width,
            height=pil_image.height,
            original_path=to_relative(upload_path),
            upload_path=to_relative(upload_path),
            overlay_path=to_relative(overlay_path),
            segmentation_map_path=to_relative(map_path),
            preview_path=to_relative(preview_path),
            mask_paths={
                label: to_relative(path) for label, path in mask_paths.items()
            },
            pixel_counts=artifacts.pixel_counts,
        )
        metadata_path = save_json(metadata.model_dump(), "segmentation_metadata")

        return SegmentResponse(
            width=pil_image.width,
            height=pil_image.height,
            original_image=pil_to_base64_png(pil_image),
            masks=mask_payload,
            overlay=overlay,
            segmentation_map=segmentation_map,
            original_path=to_relative(upload_path),
            upload_path=to_relative(upload_path),
            overlay_path=to_relative(overlay_path),
            segmentation_map_path=to_relative(map_path),
            mask_paths={
                label: to_relative(path) for label, path in mask_paths.items()
            },
            metadata_path=to_relative(metadata_path),
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Segmentation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/inpaint", response_model=ImageResponse)
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    preset_key: str | None = Form(None),
    negative_prompt: str | None = Form(None),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(6.5),
    controlnet_conditioning_scale: float = Form(0.7),
    service: InpaintingService = Depends(get_inpainting_service),
) -> ImageResponse:
    try:
        base_image = await read_image_upload(image)
        mask_image = await read_image_upload(mask)
        save_upload(base_image, "inpaint_source")
        save_upload(mask_image, "inpaint_mask")
        if preset_key:
            prompt = MATERIAL_PROMPTS.get(preset_key, prompt)
        result = service.inpaint(
            image=base_image,
            mask=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        save_output(result.image, "inpaint_result")

        return ImageResponse(image=pil_to_base64_png(result.image))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inpainting failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@router.post("/inpaint_from_class", response_model=ImageResponse)
async def inpaint_from_class(
    image: UploadFile = File(...),
    class_name: str = Form(...),
    prompt: str = Form(...),
    service: InpaintingService = Depends(get_inpainting_service),
    seg_service: SegmentationService = Depends(get_segmentation_service),
) -> ImageResponse:
    try:
        pil_image = await read_image_upload(image)
        # run segmentation to get masks
        artifacts = seg_service.segment(pil_image)
        mask = artifacts.masks.get(class_name)
        if mask is None:
            raise HTTPException(status_code=400, detail=f"Unknown class: {class_name}")
        # ensure mask is binary (white where we want to fill)
        mask_l = mask.convert("L")
        # call inpaint engine
        result = service.inpaint(image=pil_image, mask=mask_l, prompt=prompt)
        save_output(result.image, "inpaint_from_class_result")
        return ImageResponse(image=pil_to_base64_png(result.image))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inpaint-from-class failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
