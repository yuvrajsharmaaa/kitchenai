# KitchenAI

KitchenAI is an MVP for kitchen segmentation and image editing. It combines:
- A FastAPI backend for image upload, segmentation, and inpainting.
- A SegFormer segmentation pipeline.
- A Diffusers + ControlNet inpainting pipeline.
- SUN RGB-D conversion utilities for training experiments.

This README is intentionally detailed so a beginner can understand the codebase and safely make changes.

## Table of contents
- [Project overview](#project-overview)
- [Getting started](#getting-started)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [Code deep dive](#code-deep-dive)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [Future enhancements](#future-enhancements)
- [License](#license)

## Project overview

**Purpose**
- Provide a working end-to-end demo: segmentation → class selection → inpainting.
- Save all artifacts (overlay, masks, metadata) for debugging and iteration.

**Key features**
- Upload a kitchen image and get segmentation masks and overlay visualizations.
- Edit a class (e.g., wall or cabinet) by generating an inpaint mask and running ControlNet.
- Dataset conversion utilities for SUN RGB-D to support future training.

## Getting started

### Prerequisites
- Python 3.10+ (3.11 recommended)
- Git
- (Optional) CUDA-compatible GPU for faster inference
- (Optional) Hugging Face token for gated models

### Installation

```powershell
cd C:\Users\Asus\Desktop\kitchyen\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the backend

```powershell
cd C:\Users\Asus\Desktop\kitchyen\backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Open the UI at http://127.0.0.1:8001/.

## Project structure

Top-level layout:

```
backend/                  FastAPI backend + models + training helpers
frontend/                 Separate frontend repo (Next.js app)
SUNRGB/                   Raw SUN RGB-D dataset (ignored by .gitignore)
README.md                 This file
RELEASE.md                Release notes
```

Backend layout (key folders):

```
backend/app/
  ai/                     ML models, loaders, and pipelines
    inpainting/           ControlNet inpaint pipeline helpers
    segmentation/         SegFormer wrapper + postprocessing
  api/                    FastAPI routes and dependencies
  services/               Business logic for segmentation/inpainting
  utils/                  Image and file utilities
  main.py                 FastAPI app + demo HTML UI
backend/training/sunrgbd/ Dataset conversion + training scripts
```

## Architecture

Request flow:
1. A client uploads an image to `/api/segment`.
2. The backend loads a SegFormer model (cached in `ModelStore`).
3. A segmentation map and per-class masks are generated.
4. Overlay, map, masks, and metadata are saved to `backend/outputs/`.
5. The response returns base64 images and metadata paths.

Inpaint flow:
1. The client provides an image and a binary mask (white = fill area).
2. The backend prepares the mask and generates a Canny control image.
3. The Diffusers ControlNet pipeline runs and returns an edited image.

## Code deep dive

This section walks through the most important modules. Start with the routes, then follow the service layer and model wrappers.

### `backend/app/main.py`
- Initializes the FastAPI app and CORS settings.
- Serves a minimal HTML UI at `/` for uploading and editing images.
- Adds `/favicon.ico` to avoid browser 404 noise.

### `backend/app/api/routes.py`
Core API endpoints:
- `GET /api/health`: basic health check.
- `POST /api/upload`: save a raw upload and return its metadata.
- `POST /api/segment`: run SegFormer, save outputs, and return base64 images + metadata.
- `POST /api/inpaint`: run ControlNet inpainting given an image + mask.
- `POST /api/inpaint_from_class`: create a mask from segmentation output and inpaint that class.

How parts interact (example for `/api/segment`):
1. `read_image_upload` converts the file to `PIL.Image`.
2. `SegmentationService.segment` calls `SegFormerSegmenter.segment`.
3. Outputs are persisted using `save_output` and `save_json`.
4. Response is assembled using `SegmentResponse`.

### `backend/app/ai/loaders.py`
- Implements `ModelStore` to cache heavy model objects.
- Lazily loads SegFormer and Diffusers pipelines on first use.
- Uses `torch.float16` on CUDA for memory savings.

### `backend/app/ai/segmentation/segformer.py`
- Runs the processor/model and resizes logits to match the original image.
- Builds masks for `TARGET_LABELS` and counts pixels.
- Produces a colorized segmentation map and a transparent overlay.

### `backend/app/ai/inpainting/controlnet.py`
- Resizes inputs to multiples of 8 (Stable Diffusion requirement).
- Blurs/dilates the mask for smoother blending.
- Generates Canny edges to guide ControlNet.

### `backend/app/services/*.py`
- `SegmentationService` selects CUDA/CPU and runs segmentation.
- `InpaintingService` selects CUDA/CPU and runs inpainting.

### Algorithms and data structures
- **Segmentation:** argmax over per-pixel logits -> class id map.
- **Mask generation:** `prediction == label_id` per class -> binary mask.
- **Overlay:** alpha compositing with semi-transparent color layers.
- **Inpainting control:** Canny edges guide Stable Diffusion to preserve geometry.

## Configuration

Settings are defined in `backend/app/core/config.py` and read from environment variables:

- `API_PREFIX` (default: `/api`)
- `CORS_ORIGINS` (default: `http://localhost:3000`)
- `SEGFORMER_MODEL` (default: `nvidia/segformer-b0-finetuned-ade-512-512`)
- `INPAINT_MODEL` (default: `runwayml/stable-diffusion-inpainting`)
- `CONTROLNET_MODEL` (default: `lllyasviel/sd-controlnet-canny`)

Example (PowerShell):

```powershell
$env:API_PREFIX = "/api"
$env:CORS_ORIGINS = "http://localhost:3000"
$env:SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
$env:INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"
$env:CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
```

## Testing

There are no automated tests yet. Recommended basic checks:

```powershell
python -m compileall backend/app
```

Suggested future tests:
- Unit tests for `utils` and mask preprocessing.
- Integration tests that run segmentation on a tiny fixture image.

## Contributing

See CONTRIBUTING.md for a beginner-friendly contribution guide. Typical steps:
1. Fork and create a branch.
2. Make changes and add tests if relevant.
3. Open a pull request with a clear description.

## Future enhancements

- Add a proper frontend (using the `frontend/` app) with mask editing UI.
- Train a kitchen-specific segmentation model using the SUN RGB-D conversion scripts.
- Add tests and CI to cover model loading, mask generation, and API schema validation.

## License

No license file is included yet. Add a `LICENSE` file if you want to open-source the project.
