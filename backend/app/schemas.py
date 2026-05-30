from typing import Dict, List, Optional

from pydantic import BaseModel


class SegmentMetadata(BaseModel):
    model: str
    device: str
    labels: List[str]
    width: int
    height: int
    session_id: str
    original_path: str
    upload_path: str
    overlay_path: str
    segmentation_map_path: str
    preview_path: str
    mask_paths: Dict[str, str]
    pixel_counts: Optional[Dict[str, int]] = None


class SegmentResponse(BaseModel):
    width: int
    height: int
    session_id: str
    original_image: str
    masks: Dict[str, str]
    overlay: str
    segmentation_map: str
    original_path: str
    upload_path: str
    overlay_path: str
    segmentation_map_path: str
    mask_paths: Dict[str, str]
    metadata_path: str
    metadata: SegmentMetadata


class UploadResponse(BaseModel):
    width: int
    height: int
    upload_path: str


class PresetItem(BaseModel):
    key: str
    prompt: str


class PresetResponse(BaseModel):
    items: List[PresetItem]


class ImageResponse(BaseModel):
    image: str
