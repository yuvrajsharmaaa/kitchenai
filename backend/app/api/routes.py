from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.dependencies import get_inpainting_service, get_segmentation_service
from app.ai.styles_config import STYLE_CONFIG
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
from app.services.redesign import redesign_kitchen
from app.utils.files import save_json, save_output, save_upload, to_relative
from app.utils.image_io import pil_to_base64_png, read_image_upload
from app.utils.paths import get_outputs_dir
from app.utils.visualization import save_segmentation_preview
from PIL import Image

if TYPE_CHECKING:
    from app.services.inpainting import InpaintingService

settings = get_settings()
router = APIRouter(prefix=settings.api_prefix)
logger = logging.getLogger(__name__)


class RedesignRequest(BaseModel):
    session_id: str
    style_name: str
    target_classes: list[str] | None = None


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

        session_id = service.persist_session(pil_image, artifacts)

        # Convert PIL images to base64 for the HTTP response.
        mask_payload = {
            label: pil_to_base64_png(mask) for label, mask in artifacts.masks.items()
        }
        overlay = pil_to_base64_png(artifacts.overlay)
        segmentation_map = pil_to_base64_png(artifacts.segmentation_map)

        # Persist metadata so results can be replayed without re-running models.
        metadata = SegmentMetadata(
            model=settings.segmentation_model,
            device=service.device,
            labels=list(artifacts.masks.keys()),
            width=pil_image.width,
            height=pil_image.height,
            session_id=session_id,
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
            session_id=session_id,
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
        # Execute the inpainting pipeline with optional prompt overrides.
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
        # Run segmentation to derive a class-specific binary mask.
        artifacts = seg_service.segment(pil_image)
        mask = artifacts.masks.get(class_name)
        if mask is None:
            raise HTTPException(status_code=400, detail=f"Unknown class: {class_name}")
        # Ensure mask is binary (white where we want to fill).
        mask_l = mask.convert("L")
        # Call the inpaint engine using the derived mask.
        result = service.inpaint(image=pil_image, mask=mask_l, prompt=prompt)
        save_output(result.image, "inpaint_from_class_result")
        return ImageResponse(image=pil_to_base64_png(result.image))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inpaint-from-class failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/redesign")
async def redesign_endpoint(req: RedesignRequest) -> dict:
    outputs_dir = get_outputs_dir() / req.session_id
    if not outputs_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Session {req.session_id} not found"
        )

    orig_path = outputs_dir / "original.jpg"
    if not orig_path.exists():
        orig_path = outputs_dir / "original.png"
    if not orig_path.exists():
        raise HTTPException(status_code=404, detail="Original image not found")

    original_image = Image.open(orig_path).convert("RGB")

    masks: dict[str, Image.Image] = {}
    for mask_file in outputs_dir.glob("mask_*.png"):
        class_name = mask_file.stem.replace("mask_", "")
        masks[class_name] = Image.open(mask_file).convert("L")

    if not masks:
        raise HTTPException(
            status_code=422,
            detail="No masks found for this session. Re-run segmentation.",
        )

    result = redesign_kitchen(
        original_image=original_image,
        masks=masks,
        style_name=req.style_name,
        target_classes=req.target_classes,
    )

    return {
        "session_id": req.session_id,
        "style": result["style"],
        "style_label": result["style_label"],
        "before_b64": result["before_b64"],
        "after_b64": result["after_b64"],
    }


@router.get("/styles")
async def list_styles() -> dict:
    return {
        "styles": [
            {"id": key, "label": value["label"]}
            for key, value in STYLE_CONFIG.items()
        ]
    }
