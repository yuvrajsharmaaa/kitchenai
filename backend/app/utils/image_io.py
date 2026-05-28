import base64
import io
from typing import Tuple

from fastapi import UploadFile
from PIL import Image


async def read_image_upload(upload: UploadFile) -> Image.Image:
    contents = await upload.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")


def pil_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(data: str) -> Image.Image:
    raw = base64.b64decode(data.encode("utf-8"))
    return Image.open(io.BytesIO(raw)).convert("RGB")


def pil_to_bytes(image: Image.Image) -> Tuple[bytes, str]:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), "image/png"
