# Kitchen Segmentation and Editing Backend

This backend is a FastAPI MVP for kitchen image understanding and editing. It combines semantic segmentation, visualization, and optional inpainting so you can upload a kitchen photo, inspect the scene classes, and edit targeted regions locally from the browser.

The project is designed to do two related jobs:

1. Convert SUN RGB-D annotations into a clean segmentation dataset for training or fine-tuning.
2. Run a local web backend that segments kitchen images and can inpaint selected regions using a prompt.

## What the project does

At a high level, the backend does the following:

- Accepts a kitchen image upload through a local web page or API endpoint.
- Runs a pretrained SegFormer model to predict semantic classes.
- Extracts kitchen-relevant classes such as floor, wall, cabinet, countertop, furniture, and appliances.
- Builds color overlays and per-class binary masks.
- Saves the original upload and generated artifacts to disk.
- Provides an inpainting endpoint that can use a class mask or a manually supplied binary mask to edit the image.
- Includes preprocessing scripts for SUN RGB-D so you can create a dataset that is compatible with HuggingFace Transformers and SegFormer training.

## Project goals

This codebase is intentionally split into two layers:

- **Inference layer**: a FastAPI application for local testing and image editing.
- **Training data layer**: scripts that convert SUN RGB-D into a flat image/mask dataset.

That separation is important because the app can run immediately with a pretrained model, while the dataset pipeline lets you later fine-tune a stronger kitchen-specific model.

## Repository layout

```text
backend/
├── app/
│   ├── ai/
│   │   ├── loaders.py
│   │   ├── inpainting/
│   │   │   ├── config.py
│   │   │   ├── controlnet.py
│   │   │   └── prompts.py
│   │   └── segmentation/
│   │       └── segformer.py
│   ├── api/
│   │   ├── dependencies.py
│   │   └── routes.py
│   ├── core/
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── logging.py
│   │   └── security.py
│   ├── services/
│   │   ├── inpainting.py
│   │   └── segmentation.py
│   ├── storage/
│   │   ├── uploads/
│   │   └── outputs/
│   ├── utils/
│   │   ├── files.py
│   │   ├── image_io.py
│   │   ├── paths.py
│   │   └── visualization.py
│   ├── main.py
│   └── schemas.py
├── training/
│   └── sunrgbd/
│       ├── convert_sunrgbd.py
│       ├── split_dataset.py
│       ├── visualize_masks.py
│       ├── train_segformer.py
│       ├── label_mapping.py
│       └── configs/
├── outputs/
├── uploads/
├── models/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Target classes

The kitchen-focused classes used by the conversion pipeline and the app are:

- floor
- wall
- cabinet
- countertop
- furniture
- appliances

These are the classes you will see in the generated masks, segmentation overlay, and inpainting controls.

## How the backend works

### 1) Upload handling

When you upload an image, the backend validates the file, converts it to a PIL image, and stores a copy in the upload directory for traceability.

Relevant code:

- [app/api/routes.py](app/api/routes.py)
- [app/utils/image_io.py](app/utils/image_io.py)
- [app/utils/files.py](app/utils/files.py)

The upload step is intentionally separate from inference so that the original input image is preserved exactly as submitted.

### 2) Segmentation inference

The segmentation service uses HuggingFace SegFormer:

- Default model: `nvidia/segformer-b0-finetuned-ade-512-512`
- Runtime wrapper: [app/services/segmentation.py](app/services/segmentation.py)
- Model loading and caching: [app/ai/loaders.py](app/ai/loaders.py)
- Prediction logic: [app/ai/segmentation/segformer.py](app/ai/segmentation/segformer.py)

The inference flow is:

1. Load the image into memory.
2. Run the SegFormer processor to create tensor inputs.
3. Run the model forward pass.
4. Upsample logits back to the original image size.
5. Convert the top predicted label map into per-class binary masks.
6. Build a color overlay and a colorized segmentation map.
7. Return the artifacts and save them to disk.

The masks are binary images where white means the class is present and black means background.

### 3) Visualization outputs

The backend generates several visual artifacts for easy inspection:

- Original uploaded image
- Colored overlay
- Segmentation map
- One binary PNG mask per detected class
- Metadata JSON describing the run
- Optional preview grid for debugging

These are saved in `backend/outputs/` and also returned to the browser as base64 images for immediate display.

### 4) Inpainting / image editing

The editing flow uses a Stable Diffusion inpainting pipeline with ControlNet guidance.

- Model loading and caching: [app/ai/loaders.py](app/ai/loaders.py)
- Inpainting wrapper: [app/services/inpainting.py](app/services/inpainting.py)
- ControlNet image preparation: [app/ai/inpainting/controlnet.py](app/ai/inpainting/controlnet.py)
- Prompt presets: [app/ai/inpainting/prompts.py](app/ai/inpainting/prompts.py)

The inpainting flow is:

1. Receive the original image and a mask.
2. Resize both to a Stable Diffusion-friendly size.
3. Convert the mask to grayscale.
4. Create a Canny edge control image from the original image.
5. Feed the prompt, mask, and control image into the diffusion pipeline.
6. Save and return the edited result.

The app supports two edit modes:

- **Manual edit**: upload your own binary mask.
- **Class edit**: click `Edit this class` under a segmentation mask and let the backend reuse the predicted class mask.

## File-by-file guide

### [app/main.py](app/main.py)

Creates the FastAPI app, enables CORS, mounts the API router, and serves a simple browser UI at `/`.

This file is the entry point when you run:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

It also handles `/favicon.ico` so the browser does not spam the console with a 404.

### [app/api/routes.py](app/api/routes.py)

Defines the public API endpoints:

- `GET /api/health`
- `GET /api/presets`
- `POST /api/upload`
- `POST /api/segment`
- `POST /api/inpaint`
- `POST /api/inpaint_from_class`

This is the main orchestration layer. It connects the web request to the services, persists artifacts, and builds the response model.

### [app/api/dependencies.py](app/api/dependencies.py)

Creates service objects for dependency injection. This keeps route handlers small and makes the code easier to test.

### [app/services/segmentation.py](app/services/segmentation.py)

Wraps the SegFormer segmenter and exposes a simple `segment(image)` method.

### [app/services/inpainting.py](app/services/inpainting.py)

Wraps the inpainting pipeline and exposes a simple `inpaint(...)` method.

### [app/ai/loaders.py](app/ai/loaders.py)

Central model cache. It loads and stores the heavy HuggingFace objects so they do not need to be recreated on every request.

This reduces latency after the first request.

### [app/ai/segmentation/segformer.py](app/ai/segmentation/segformer.py)

Implements segmentation post-processing.

Responsibilities:

- Run the model.
- Convert logits into predicted class IDs.
- Build class-specific binary masks.
- Colorize the overlay.
- Produce a segmentation map image.

### [app/ai/inpainting/controlnet.py](app/ai/inpainting/controlnet.py)

Prepares the image and mask for diffusion inpainting.

Responsibilities:

- Resize inputs to valid dimensions.
- Blur and dilate the mask if requested.
- Create a Canny edge image for structure guidance.
- Call the loaded ControlNet inpainting pipeline.

### [app/core/config.py](app/core/config.py)

Loads environment variables and defines runtime settings such as:

- API prefix
- CORS origins
- SegFormer model name
- Inpainting model name
- ControlNet model name

### [app/core/constants.py](app/core/constants.py)

Defines the target labels and display colors used in overlays and visualization.

### [app/utils/files.py](app/utils/files.py)

Handles file saving for uploads, outputs, and metadata JSON.

### [app/utils/image_io.py](app/utils/image_io.py)

Reads uploaded image files into PIL objects and converts PIL images to base64 PNG strings.

### [app/utils/visualization.py](app/utils/visualization.py)

Builds visual previews so you can inspect the original image, overlay, and segmentation result side by side.

### [app/schemas.py](app/schemas.py)

Defines the Pydantic response models used by the API.

These models keep the API response shape explicit and consistent.

## SUN RGB-D conversion pipeline

The dataset conversion tools live in:

```text
backend/training/sunrgbd/
```

They are used to convert the original SUN RGB-D annotation structure into a flat dataset that is easy to train with HuggingFace and SegFormer.

### Source data location

Your extracted SUN RGB-D folder is expected here:

```text
C:\Users\Asus\Desktop\kitchyen\SUNRGB\SUNRGBD\SUNRGBD
```

### Conversion flow

The converter reads the original SUN RGB-D annotation indexes, rasterizes the labels into PNG masks, and copies the aligned RGB images into a training-friendly structure.

The main outputs are:

```text
dataset/
├── images/
├── masks/
└── metadata/
```

### Metadata files

The metadata directory can include:

- `dataset_info.json`
- `label_map.json`
- `samples.csv`
- `samples.jsonl`
- train/validation split files

### Training scripts

- `convert_sunrgbd.py`: converts the raw SUN RGB-D annotations into RGB/mask pairs.
- `split_dataset.py`: creates reproducible train/validation splits.
- `visualize_masks.py`: renders RGB, mask, and overlay previews.
- `train_segformer.py`: fine-tunes SegFormer on the generated dataset.
- `label_mapping.py`: defines label aliases, class IDs, and color mappings.

### Training class IDs

The training pipeline uses a stable ID mapping such as:

- `0`: background
- `1`: floor
- `2`: wall
- `3`: cabinet
- `4`: countertop
- `5`: furniture
- `6`: appliances

This mapping is written to the dataset metadata so the training and inference code agree on labels.

## Running the backend locally

### 1) Create and activate a virtual environment

From the repository root:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

If you are using a fresh machine, the first run may download large HuggingFace model weights.

### 3) Start the server

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

### 4) Open the UI

Go to:

```text
http://127.0.0.1:8001/
```

### 5) Open the docs

Go to:

```text
http://127.0.0.1:8001/docs
```

## API usage

### Health check

```powershell
curl http://127.0.0.1:8001/api/health
```

Expected response:

```json
{"status":"ok"}
```

### Segment an image

```powershell
curl -X POST "http://127.0.0.1:8001/api/segment" ^
  -F "image=@C:\path\to\kitchen.jpg"
```

This returns:

- original image as base64 PNG
- overlay as base64 PNG
- segmentation map as base64 PNG
- per-class masks as base64 PNGs
- response metadata including saved file paths

### Manual inpaint

```powershell
curl -X POST "http://127.0.0.1:8001/api/inpaint" ^
  -F "image=@C:\path\to\kitchen.jpg" ^
  -F "mask=@C:\path\to\mask.png" ^
  -F "prompt=replace tiles with white marble"
```

### Inpaint by class

```powershell
curl -X POST "http://127.0.0.1:8001/api/inpaint_from_class" ^
  -F "image=@C:\path\to\kitchen.jpg" ^
  -F "class_name=wall" ^
  -F "prompt=replace wall with soft beige textured tiles"
```

## What should not be committed

These files and folders are generated locally and should stay out of git:

- `backend/.venv/`
- `backend/uploads/`
- `backend/outputs/`
- `backend/storage/uploads/`
- `backend/storage/outputs/`
- `backend/models/` if you download or cache model files there
- `backend/training/sunrgbd/dataset/`
- `backend/training/sunrgbd/dataset_*`
- `backend/training/sunrgbd/outputs/`
- `backend/app/__pycache__/`
- `backend/app/**/__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `.env` files containing local secrets
- frontend build folders like `frontend/.next/`

The repo-level and backend-level `.gitignore` files were updated to exclude these paths.

## Why the project needs generated artifacts ignored

The backend creates many files during normal use:

- Uploaded input images
- Segmentation previews
- Binary masks
- Inpainted results
- Metadata JSON files
- Dataset conversion outputs
- Temporary cache files from HuggingFace and PyTorch

These files are useful locally, but they do not belong in source control because they are large, machine-specific, or reproducible from the scripts.

## Common workflow

### If you want to test segmentation quickly

1. Start the backend.
2. Open the local UI.
3. Upload a kitchen image.
4. Run segmentation.
5. Inspect the returned overlay and class masks.

### If you want to edit a specific region

1. Run segmentation first.
2. Pick the mask for the class you want to modify.
3. Enter an edit prompt.
4. Click `Edit this class` or upload a manual binary mask.
5. Wait for the inpaint result to appear.

### If you want to build a dataset

1. Run `convert_sunrgbd.py`.
2. Run `split_dataset.py`.
3. Inspect with `visualize_masks.py`.
4. Fine-tune with `train_segformer.py`.

## Troubleshooting

### The browser shows `404` for `favicon.ico`

That is harmless. The backend now responds to favicon requests, so you should not see it after reload.

### The first inpaint request takes a long time

This is expected. The diffusion pipeline is large and may need to download weights before it can run.

### Inpaint results look wrong

Usually one of these is the cause:

- The mask is not binary enough.
- The mask covers too much or too little.
- The prompt is too vague.
- The image is too large for the machine.

### Segmentation masks do not match the expected classes

The pretrained SegFormer model is generic. For better kitchen-specific results, fine-tune on your SUN RGB-D conversion dataset.

### Output files keep appearing in git status

Make sure you are using the updated `.gitignore` files and that the files live under the ignored paths listed above.

## Suggested next steps

- Fine-tune SegFormer on the SUN RGB-D derived kitchen dataset.
- Replace the generic inpainting model with a kitchen-specific prompt workflow.
- Add a brush-based mask editor in the frontend.
- Add download buttons for masks, overlays, and metadata.
- Add a training config file and evaluation script for SegFormer fine-tuning.

## Notes on version control

This workspace was not initialized as a git repository when these changes were made, so there is nothing to commit yet. Once you initialize git, the updated ignore files will keep generated data out of the repository.

