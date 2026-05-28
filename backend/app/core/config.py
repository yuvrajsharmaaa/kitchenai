from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    api_prefix: str
    cors_origins: list[str]
    segmentation_model: str
    inpaint_model: str
    controlnet_model: str


def get_settings() -> Settings:
    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    origin_list = [origin.strip() for origin in origins.split(",") if origin.strip()]
    return Settings(
        api_prefix=os.getenv("API_PREFIX", "/api"),
        cors_origins=origin_list,
        segmentation_model=os.getenv(
            "SEGFORMER_MODEL", "nvidia/segformer-b0-finetuned-ade-512-512"
        ),
        inpaint_model=os.getenv(
            "INPAINT_MODEL", "runwayml/stable-diffusion-inpainting"
        ),
        controlnet_model=os.getenv(
            "CONTROLNET_MODEL", "lllyasviel/sd-controlnet-canny"
        ),
    )
