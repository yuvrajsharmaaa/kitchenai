# KitchenAI — Deep Project README

This README explains the KitchenAI repository in exhaustive detail so a beginner can understand, run, and modify the project. It documents architecture, data flow, models, services, deployment, development tips, and troubleshooting.

Table of contents
- Project overview
- Quick start (run locally)
- Architecture and design decisions
- File and folder map (detailed)
- Backend: structure, key modules, and flow
- API endpoints and expected inputs/outputs
- Models used: SegFormer and Diffusers/ControlNet (details & tuning)
- SUN RGB-D conversion & training helpers
- Outputs, artifacts, and storage conventions
- Development and debugging checklist
- How to make common changes (walkthroughs)
- Performance, GPU/CPU considerations, and tips
- Tests, CI, and contribution guidance
- Appendix: common errors and solutions

=== PROJECT OVERVIEW ===

KitchenAI is an experimental MVP for kitchen segmentation and image editing. It provides:

- A FastAPI backend that accepts image uploads and returns segmentation maps and overlay visualizations.
- An inference pipeline using a pre-trained SegFormer model for semantic segmentation.
- An inpainting pipeline powered by Diffusers + ControlNet to do image edits guided by segmentation classes (e.g., remove/replace wall, countertop).
- Utilities and scripts for preparing the SUN RGB-D dataset for training and experimentation.

Goals:
- Provide a working end-to-end demo for segmentation → class selection → inpainting.
- Keep source code modular so models, processors, and UI can be improved independently.

=== QUICK START (DEV) ===

Prerequisites:
- Python 3.10+ (3.11 recommended)
- Git
- (Optional but recommended) GPU with CUDA and a compatible PyTorch build
- A Hugging Face token if you access gated models

Install and run (example using `venv`):

```powershell
cd C:\Users\Asus\Desktop\kitchyen\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Open the demo UI at: http://127.0.0.1:8001/ (the backend serves a simple HTML UI for testing uploads and actions).

Notes:
- The first inpaint request may take many minutes if model weights must be downloaded. Keep an eye on the terminal logs.
- Avoid committing `backend/outputs/`, `backend/models/`, and the raw `SUNRGB/` data to Git.

=== ARCHITECTURE & DESIGN ===

High-level components:
- `backend/app/main.py`: FastAPI app and simple root UI.
- `backend/app/api/routes.py`: HTTP endpoints that orchestrate uploads, segmentation, and inpainting.
- `backend/app/ai/`:
  - `loaders.py`: model loader and caching (`ModelStore`) to avoid repeated model loads.
  - `segmentation/segformer.py`: segmentation pipeline wrapper that runs the processor + model and produces masks, overlay, and metadata.
  - `inpainting/controlnet.py`: prepares inpainting mask, creates a Canny control map, and invokes the Diffusers pipeline.
- `backend/app/services/`: high-level services that call loaders and run end-to-end work for routing.
- `backend/app/utils/`: helpers for file I/O, image conversion, path normalization, and visualization.
- `backend/outputs/` and `backend/uploads/`: runtime artifacts and uploads (ignored by .gitignore).

Key design choices:
- Model caching: instantiate heavy models once and reuse them using `ModelStore`.
- Minimal UI served by FastAPI for quick testing — the frontend can be separated into a dedicated app later.
- Persistence of outputs and JSON metadata for reproducibility and audit.

=== FILE & FOLDER MAP (DETAILED) ===

Root highlights (relevant for contributors):
- `backend/` — backend server, models, training helpers.
  - `app/` — main FastAPI application and modules
    - `ai/` — model loaders & pipelines
      - `inpainting/` — controlnet inpaint pipeline, prompt templates
      - `segmentation/` — segformer wrapper & postprocessing
    - `api/` — `routes.py` exposes endpoints; `dependencies.py` contains DI utilities
    - `services/` — business logic for segmentation & inpainting
    - `utils/` — `files.py`, `image_io.py`, `paths.py`, `visualization.py`
  - `training/sunrgbd/` — scripts to convert SUN RGB-D and prepare datasets
- `frontend/` — separate git repo inside tree (treated as submodule-like). Contains a Next.js app for advanced UI.

File-by-file quick reference (editors will appreciate):
- `backend/app/main.py`: register routes, include CORS, serve demo HTML. Edit to change host/port, static assets, or add more routes.
- `backend/app/api/routes.py`: endpoints:
  - `POST /api/segment`: expects a multipart file upload; returns segmentation maps, overlay base64, paths saved to `backend/outputs/`, and a JSON metadata file.
  - `POST /api/inpaint`: expects `image` + `mask` files or base64; uses inpainting pipeline to produce edited image(s).
  - `POST /api/inpaint_from_class`: convenience endpoint to create binary masks from segmentation classes and run inpaint.
  - `GET /api/health`: returns OK.
- `backend/app/ai/loaders.py`: implement lazy loading. If you want to swap models, modify loader functions to return alternative checkpoints.
- `backend/app/ai/segmentation/segformer.py`: contains post-processing code to generate per-class binary masks, overlay generation, and `pixel_counts` metadata.
- `backend/app/ai/inpainting/controlnet.py`: preprocess mask (white=fill area), generate Canny map for control net conditioning, call the `StableDiffusionControlNetInpaintPipeline` with guidance, and return output images.
- `backend/app/utils/files.py`: `save_output()` and `save_json()` to store artifacts under `backend/outputs/`.

=== API DETAILS & SAMPLES ===

Segment endpoint example (curl):

```bash
curl -X POST "http://127.0.0.1:8001/api/segment" -F "file=@kitchen.jpg"
```

Response (JSON):
- `original_image`: base64 of original
- `overlay_path`, `segmentation_map_path`, `mask_paths`: server filesystem paths (timestamped)
- `metadata_path`: JSON on disk containing labels, pixel counts, and mapping info

Inpaint endpoint notes:
- Send `image` and `mask` (mask white = area to be inpainted). Coordinates and size must match. The `mask` is preprocessed to the model's expected format.
- `inpaint_from_class` will internally create the binary mask for a target class (e.g., `wall`) and run the inpainting pipeline.

=== MODELS: DETAILS & TUNING ===

Segmentation (SegFormer)
- Checkpoint used (by default): `nvidia/segformer-b0-finetuned-ade-512-512` (configured in `loaders.py`)
- Processor: the corresponding feature extractor/processor resizes input images to the model expected resolution (typically square, e.g., 512x512). The loader returns both processor and model in a bundle.
- Output handling: post-process logits to generate argmax predictions; resize predictions to original image size using nearest-neighbor to maintain class id integrity.

Inpainting (Diffusers + ControlNet)
- We use a Stable Diffusion inpaint pipeline adapted with a ControlNet conditioning map (Canny edges derived from the original image). The code expects the `StableDiffusionControlNetInpaintPipeline` class and an available controlnet checkpoint.
- Memory and time considerations: these pipelines are large (~several GB) — prefer GPU with >8GB VRAM. For local development without GPU, expect very slow inference.

Customizing models:
- To swap segmentation model: change the model name in `loaders.py` and ensure the processor used is compatible.
- To reduce memory: use a smaller diffusion backbone or a lower `height`/`width` in inference. You can also run the Diffusers pipeline in `torch.float16` where supported.

=== SUN RGB-D DATASET & TRAINING HELPERS ===

The training scripts are in `backend/training/sunrgbd/` and provide utilities to convert the raw SUNRGB dataset into a simplified structure used for training segmentation models.

Key scripts:
- `convert_sunrgbd.py`: parse SUNRGB annotations and export images and masks into `dataset/images/` and `dataset/masks/` with matching filenames.
- `label_mapping.py`: map SUNRGBD class ids to your target taxonomy (kitchen-focused labels such as `floor`, `wall`, `cabinet`, `countertop`). Update `configs/label_map.json` to match your classes.
- `prepare_dataset.py` and `split_dataset.py`: create training/validation splits and optional TFRecord or preprocessed caches.
- `train_segformer.py`: example training harness that uses Hugging Face Transformers/Trainer or a PyTorch loop. It is a starting point — you will likely adapt hyperparameters and dataset transforms for best kitchen performance.

Dataset layout expected by training scripts:

```
backend/training/sunrgbd/dataset/
  images/
  masks/
  metadata.json
```

=== OUTPUTS & ARTIFACTS ===

- `backend/uploads/`: saved user uploads (ignored from Git)
- `backend/outputs/`: saved inference results with timestamped filenames and a metadata JSON file summarizing the run. This includes overlay PNGs, segmentation maps, per-class masks, and inpainted outputs.

Naming conventions: filenames include a timestamp prefix and a descriptive suffix (e.g., `20260528_144500_segmentation_overlay.png`). Metadata JSON includes relative paths so the frontend can load images easily.

=== DEVELOPMENT & DEBUGGING CHECKLIST ===

1. Start with a fresh Python venv and install `requirements.txt`.
2. If segmentation outputs are blank or wrong: check image resizing logic in `segformer.py` and confirm the class mapping in `constants.py`.
3. If inpainting fails with OOM: reduce image size, enable CPU fallback, or switch to a smaller pipeline.
4. If UI returns `404 Not Found` for root: ensure `app.main` has the `index()` route and uvicorn is started against `app.main:app`.
5. Frequent logs:
   - Diffusers will log downloads when weights are missing — this is expected on first run.

=== HOW TO MAKE COMMON CHANGES (WALKTHROUGHS) ===

Change target segmentation labels (e.g., add `oven`):
1. Edit `backend/app/core/constants.py` to include your new `TARGET_LABELS` and mappings.
2. Update `training/sunrgbd/configs/label_map.json` if training.
3. Rerun training or adapt postprocessing in `segformer.py` to handle the new label.

Swap segmentation backbone:
1. In `ai/loaders.py` change the model ID to a new SegFormer or other semantic segmentation model.
2. Verify the processor signature and adjust preprocessing sizing in the segmentation code.
3. Run a few inference checks with the demo UI.

Tune inpainting prompts and strength:
- Prompts are in `ai/inpainting/prompts.py`. Edit templates and default negative prompts.
- Adjust guidance scale, steps, and the controlnet conditioning strength inside `inpainting/controlnet.py`.

=== PERFORMANCE & DEPLOYMENT NOTES ===

- On GPU: install a matching CUDA-enabled PyTorch and set `torch.backends.cudnn.benchmark = True` for perf.
- Model caching: ensure only one process loads models (or use server-level locking) to avoid duplicating VRAM usage.
- For production: consider containerizing the service and mounting a pre-downloaded model cache volume to avoid repeated downloads.

=== TESTS & CI ===

This repo does not include automated tests yet. Suggested tests:
- Unit: small tests for `utils` functions, image IO, and mask preprocessing.
- Integration: run segmentation inference on a tiny fixture image and validate outputs exist.
- CI: add a GitHub Actions workflow to lint, run unit tests, and optionally build Docker images.

=== CONTRIBUTING & PROJECT GUIDELINES ===

- Respect `.gitignore` — do not commit large datasets or model weights.
- Use branches named `feature/*`, `fix/*`, `docs/*`.
- For large model changes, open an issue describing the motivation so reviewers can evaluate resource impacts.

=== APPENDIX: COMMON ERRORS & SOLUTIONS ===

- Diffusers/download delays: patience on first run. If interrupted, delete partial caches under `%USERPROFILE%\.cache\huggingface` and retry.
- OOM on GPU: lower `height`/`width`, switch to CPU for debugging, or reduce batch size.
- `Not Found` on root: check `backend/app/main.py` includes route for `/`.

---

If you'd like, I can also generate a smaller `CONTRIBUTING.md`, CI workflow, or inline code comments for the three most critical files. Tell me which next.
