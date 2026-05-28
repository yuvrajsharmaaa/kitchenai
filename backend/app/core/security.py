from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class UploadLimits:
    max_bytes: int


def get_upload_limits() -> UploadLimits:
    return UploadLimits(max_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))))
