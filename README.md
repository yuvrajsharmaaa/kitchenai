# 🍳 KitchenAI — AI-Powered Kitchen Redesign Tool

> Upload a photo of your kitchen. Pick a style. Watch it transform.

KitchenAI uses computer vision and generative AI to segment a kitchen photo into its structural regions — floor, cabinets, backsplash, countertop — and then applies realistic material and style transformations to those regions. The output is a side-by-side before/after comparison of the kitchen with the chosen design style applied.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Getting Started](#2-getting-started)
3. [Project Structure and Architecture](#3-project-structure-and-architecture)
4. [How It Works — In Depth](#4-how-it-works--in-depth)
5. [Available Styles](#5-available-styles)
6. [Usage Guide](#6-usage-guide)
7. [API Reference](#7-api-reference)
8. [Configuration](#8-configuration)
9. [Hardware Requirements](#9-hardware-requirements)
10. [Contributing](#10-contributing)
11. [Roadmap & Future Enhancements](#11-roadmap--future-enhancements)
12. [Known Issues](#12-known-issues)
13. [License](#13-license)
14. [Contact & Support](#14-contact--support)

---

## 1. Project Overview

### What Problem Does This Solve?

Hiring an interior designer, ordering material samples, and physically visualising what a kitchen redesign would look like is slow and expensive. KitchenAI allows anyone to take a photo of their existing kitchen and instantly see what it would look like with a completely different material palette — marble countertops, dark luxury cabinets, rustic wood floors — in seconds, for free, running entirely on their own machine.

### Key Features

- **Semantic segmentation** — automatically detects and labels kitchen regions (floor, cabinets, backsplash, countertops) in any uploaded photo
- **Four curated design styles** — Italian Marble, Wooden Rustic, Modern White, and Dark Luxury
- **Realistic image generation** — uses Stable Diffusion inpainting to synthesise photorealistic material changes that respect the original room's geometry, lighting, and perspective
- **Before/after comparison** — returns both the original and the redesigned image for easy comparison
- **Fully local** — no cloud API required; everything runs on your own machine
- **Simple REST API** — designed to be consumed by any frontend (the included Next.js frontend or your own)

### Who Is This For?

- **Homeowners** who want to visualise kitchen renovation ideas before committing
- **Interior design students** exploring how material changes affect a space
- **Developers** building property tech, real estate, or home improvement applications
- **AI/ML engineers** interested in combining semantic segmentation with generative inpainting

---

## 2. Getting Started

### 2.1 Prerequisites

Before installing, ensure you have the following:

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| Node.js | 18+ | For the frontend |
| npm | 9+ | Comes with Node.js |
| Git | Any | For cloning |
| CUDA (GPU) | Optional | Required for fast generation (~30 sec); CPU works but is very slow (~15 min) |

> **No Hugging Face account required** for the default models. They are publicly available and download automatically on first run.

### 2.2 Clone the Repository

```bash
git clone https://github.com/yuvrajsharmaaa/kitchenai.git
cd kitchenai
```

### 2.3 Backend Setup

```bash
# Step 1: Navigate to the backend directory
cd backend

# Step 2: Create and activate a Python virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS / Linux:
source .venv/bin/activate

# Step 3: Install all Python dependencies
pip install -r requirements.txt

# Step 4: Install any missing dependencies for the generation pipeline
pip install "diffusers>=0.27.0" accelerate safetensors
```

### 2.4 Frontend Setup

```bash
# In a separate terminal, navigate to the frontend directory
cd frontend

# Install Node.js dependencies
npm install
```

### 2.5 Running the Project

**Terminal 1 — Start the backend API:**

```bash
cd backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

**Terminal 2 — Start the frontend:**

```bash
cd frontend
npm run dev
```

Now open your browser:
- **Frontend (user interface):** http://localhost:3000
- **Backend API docs (interactive):** http://127.0.0.1:8001/docs
- **Backend health check:** http://127.0.0.1:8001/api/health

> **First-run model downloads:** On the very first run, the system will automatically download two AI models from Hugging Face. This requires an internet connection and approximately 4.5 GB of disk space. The downloads are cached and do not repeat on subsequent runs.
>
> | Model | Size | Purpose |
> |---|---|---|
> | `nvidia/segformer-b0-finetuned-ade-512-512` | ~300 MB | Kitchen region detection |
> | `runwayml/stable-diffusion-inpainting` | ~4.2 GB | Style generation |

---

## 3. Project Structure and Architecture

### 3.1 Directory Layout

```
kitchenai/
│
├── backend/                        ← Python FastAPI server + AI models
│   ├── app/
│   │   ├── ai/                     ← All AI/ML model code
│   │   │   ├── segmentation/
│   │   │   │   └── segformer.py    ← Detects kitchen regions in a photo
│   │   │   ├── inpainting/
│   │   │   │   └── sd_inpaint.py   ← Generates styled images via Stable Diffusion
│   │   │   ├── styles_config.py    ← Defines the 4 design styles and their prompts
│   │   │   └── loaders.py          ← Loads and caches AI models in memory
│   │   │
│   │   ├── api/
│   │   │   └── routes.py           ← HTTP endpoints: /segment, /redesign, /styles
│   │   │
│   │   ├── core/
│   │   │   └── config.py           ← App settings and environment variable reader
│   │   │
│   │   ├── services/
│   │   │   ├── segmentation.py     ← Business logic for running segmentation
│   │   │   └── redesign.py         ← Business logic for running style application
│   │   │
│   │   ├── utils/                  ← Image helpers, file I/O utilities
│   │   │   └── main.py             ← App entry point; starts FastAPI, adds CORS
│   │   │
│   ├── outputs/                    ← Auto-created at runtime; stores session files
│   │   └── {session_id}/
│   │       ├── original.jpg        ← The uploaded kitchen image
│   │       ├── mask_floor.png      ← Binary mask for the floor region
│   │       ├── mask_cabinet.png    ← Binary mask for the cabinet region
│   │       ├── mask_backsplash.png ← Binary mask for the backsplash region
│   │       └── mask_countertop.png ← Binary mask for the countertop region
│   │
│   └── requirements.txt            ← Python package list
│
├── frontend/                       ← Next.js web application
│   └── (standard Next.js layout)
│
├── README.md                       ← This file
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

### 3.2 Core Components and How They Interact

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                             │
│   User uploads image → calls /api/segment                  │
│   User picks a style → calls /api/redesign                 │
│   Displays before/after images from response               │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP (JSON + base64 images)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                          │
│                                                             │
│   routes.py  ←──────────────────── Receives HTTP requests  │
│       │                                                     │
│       ├──► SegmentationService ──► SegFormer model         │
│       │         saves masks to outputs/{session_id}/       │
│       │                                                     │
│       └──► RedesignService ──────► SD Inpainting model     │
│                 loads masks, runs pipeline, composites      │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    AI MODELS (local)                        │
│                                                             │
│  SegFormer (segmentation) ── identifies regions by pixel   │
│  Stable Diffusion Inpaint ── generates new material pixels │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow

Here is the complete journey of a kitchen image through the system:

```
1. USER UPLOADS IMAGE
   └─► POST /api/segment
       └─► SegmentationService.segment()
           └─► SegFormerSegmenter.segment()
               ├─ Resizes image to 512×512
               ├─ Runs SegFormer model → pixel-level class predictions
               ├─ Maps predictions to: floor / cabinet / backsplash / countertop
               ├─ Saves original.jpg and mask_*.png to outputs/{session_id}/
               └─ Returns: session_id, overlay image (base64), class masks

2. USER SELECTS STYLE
   └─► POST /api/redesign { session_id, style_name }
       └─► RedesignService.redesign_kitchen()
           ├─ Loads original.jpg from outputs/{session_id}/
           ├─ Loads all mask_*.png files
           ├─ Looks up style prompt from styles_config.py
           ├─ Combines masks into one composite mask
           └─► sd_inpaint.run_sd_inpaint()
               ├─ Resizes to SD-compatible dimensions (multiple of 8)
               ├─ Runs Stable Diffusion Inpainting pipeline
               ├─ Composites result onto original image
               └─ Returns: before_b64, after_b64

3. FRONTEND DISPLAYS
   └─ Side-by-side before/after images
```

---

## 4. How It Works — In Depth

This section explains the technology behind each step in plain language.

### 4.1 What Is Semantic Segmentation?

Semantic segmentation is the process of labelling every single pixel in an image with a category. Think of it like colouring in a photograph, but done by an AI — every pixel belonging to the floor gets coloured one colour, every pixel belonging to a cabinet gets another colour, and so on.

KitchenAI uses a model called **SegFormer**, developed by NVIDIA. It was trained on the **ADE20K dataset**, a large collection of indoor scenes annotated with 150 object categories. When given a kitchen photo, SegFormer produces a map where each pixel has an assigned class ID.

The relevant ADE20K class IDs for kitchens are:

| Region | ADE20K Label ID | Notes |
|---|---|---|
| Floor | 3 | Kitchen and general floor |
| Wall / Backsplash | 0 | Full wall — MVP uses lower portion |
| Cabinet | 24 | Kitchen cabinetry |
| Countertop | 46 | Counter surfaces |
| Table / Island | 15 | Kitchen islands if present |

After segmentation, the system creates a **binary mask** for each class. A binary mask is a black-and-white image the same size as the original photo, where white pixels mean "this region" and black pixels mean "not this region". These masks are the key to targeted material replacement.

### 4.2 What Is Inpainting?

Inpainting is the process of filling in a region of an image with AI-generated content. It was originally used to remove unwanted objects from photos (like filling in where a person stood). Here, we use it creatively — instead of removing something, we're replacing kitchen surfaces with new materials.

**Stable Diffusion Inpainting** takes three inputs:
1. The original image
2. A mask (which areas to change)
3. A text prompt (what the new content should look like)

The model then generates new pixels for the masked area that match the prompt, while keeping everything else intact. Because Stable Diffusion was trained on millions of real photographs, it has a very good understanding of lighting, texture, perspective, and how surfaces interact with the rest of a scene.

### 4.3 The Mask Pipeline

Before passing masks to the inpainter, the system processes them to avoid hard edges that would look unrealistic:

```
Raw binary mask (0 or 255 per pixel)
        │
        ▼
Max Filter (5px) — dilates the mask slightly to avoid missing edge pixels
        │
        ▼
Gaussian Blur (radius 2) — softens the boundary for smooth blending
        │
        ▼
Composite mask (used as alpha channel for blending)
```

After inpainting, the generated image is blended back onto the original using the mask as an alpha channel — so regions outside the mask are always taken directly from the original photo, guaranteeing lighting and geometry are preserved.

### 4.4 Style Prompts

Each of the four design styles is backed by a carefully crafted **text prompt** that describes exactly what the surfaces should look like after transformation. The prompt system includes:

- **Positive prompt** — what to generate (e.g., "Italian Carrara marble with grey veining, polished stone, elegant, interior photography")
- **Negative prompt** — what to avoid generating (e.g., "cartoon, blurry, low quality, watermark")
- **Strength** — how much the original image influences the result (0.0 = unchanged, 1.0 = completely new)
- **Guidance scale** — how closely to follow the prompt (higher = more literal)
- **Inference steps** — how many denoising steps to run (more = higher quality, slower)

### 4.5 Model Loading and Caching

Loading AI models is slow (10–30 seconds). The `ModelStore` class in `loaders.py` implements **lazy loading with caching** — it loads a model only on the first request that needs it and keeps it in memory for all subsequent requests. This means the first API call is slow, but all later calls are fast.

On CUDA (GPU), models are loaded in `float16` precision (half the memory of `float32`), which nearly halves VRAM usage with negligible quality loss.

---

## 5. Available Styles

| Style ID | Display Name | Description |
|---|---|---|
| `italian_marble` | 🏛️ Italian Marble | White Carrara marble with grey veining; polished, luxury feel |
| `wooden_rustic` | 🪵 Wooden Rustic | Warm oak grain cabinets, reclaimed wood flooring; farmhouse warmth |
| `modern_white` | 🤍 Modern White | Clean white lacquer surfaces, Scandinavian minimalism |
| `dark_luxury` | 🖤 Dark Luxury | Matte black cabinets, dark marble countertops; dramatic and bold |

To see styles at runtime:

```bash
curl http://127.0.0.1:8001/api/styles
```

---

## 6. Usage Guide

### 6.1 Using the Web Interface

1. Open http://localhost:3000 in your browser
2. Click **Upload Image** and select a kitchen photo (JPG or PNG)
3. Click **Segment** — the system will identify and highlight the different kitchen regions
4. Review the overlay showing detected regions (floor, cabinets, backsplash, countertop)
5. Select one of the four design style buttons
6. Click **Apply Style** — generation takes 15–90 seconds depending on your hardware
7. View the before/after comparison

### 6.2 Using the API Directly (for developers)

**Step 1 — Segment a kitchen image:**

```bash
curl -X POST "http://127.0.0.1:8001/api/segment" \
  -F "file=@/path/to/kitchen.jpg"
```

Response:
```json
{
  "session_id": "a3f1b2c4",
  "overlay_b64": "...",
  "masks": {
    "floor": "...",
    "cabinet": "...",
    "backsplash": "...",
    "countertop": "..."
  }
}
```

**Step 2 — Apply a style:**

```bash
curl -X POST "http://127.0.0.1:8001/api/redesign" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "a3f1b2c4",
    "style_name": "italian_marble"
  }'
```

Response:
```json
{
  "session_id": "a3f1b2c4",
  "style": "italian_marble",
  "style_label": "Italian Marble",
  "before_b64": "<base64-encoded-JPEG>",
  "after_b64": "<base64-encoded-JPEG>"
}
```

**Step 3 — Decode and display the images:**

```python
import base64, io
from PIL import Image
after_bytes = base64.b64decode(response["after_b64"])
Image.open(io.BytesIO(after_bytes)).save("redesigned_kitchen.jpg")
```

### 6.3 Applying Styles Only to Specific Regions

You can limit which surfaces get restyled by passing `target_classes`:

```bash
curl -X POST "http://127.0.0.1:8001/api/redesign" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "a3f1b2c4",
    "style_name": "italian_marble",
    "target_classes": ["countertop", "backsplash"]
  }'
```

This will only apply the marble style to the countertop and backsplash, leaving floors and cabinets unchanged.

### 6.4 Expected Input / Output

| Parameter | Details |
|---|---|
| Input image format | JPG, PNG |
| Recommended resolution | 512×512 to 1024×768; higher = slower |
| Output format | Base64-encoded JPEG |
| Output resolution | Same as input |
| Generation time (GPU) | ~15–90 seconds |
| Generation time (CPU) | ~10–20 minutes |

---

## 7. API Reference

### `GET /api/health`
Returns a simple health check confirming the server is running.

**Response:** `{ "status": "ok" }`

---

### `GET /api/styles`
Returns the list of available design styles.

**Response:**
```json
{
  "styles": [
    { "id": "italian_marble", "label": "Italian Marble" },
    { "id": "wooden_rustic",  "label": "Wooden Rustic" },
    { "id": "modern_white",   "label": "Modern White" },
    { "id": "dark_luxury",    "label": "Dark Luxury" }
  ]
}
```

---

### `POST /api/segment`
Segments a kitchen image into labelled regions.

**Request:** `multipart/form-data`
- `file` (required) — image file (JPG or PNG)

**Response:** JSON with `session_id`, `overlay_b64`, per-class mask images

---

### `POST /api/redesign`
Applies a design style to a previously segmented image.

**Request:** JSON body
```json
{
  "session_id": "string (required)",
  "style_name": "italian_marble | wooden_rustic | modern_white | dark_luxury (required)",
  "target_classes": ["floor", "cabinet", "backsplash", "countertop"] 
}
```

**Response:** JSON with `before_b64` and `after_b64` (both base64-encoded JPEG images)

---

## 8. Configuration

All settings are defined in `backend/app/core/config.py` and can be overridden with environment variables.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_PREFIX` | `/api` | URL prefix for all API routes |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed frontend origins (comma-separated for multiple) |
| `SEGFORMER_MODEL` | `nvidia/segformer-b0-finetuned-ade-512-512` | HuggingFace model ID for segmentation |
| `INPAINT_MODEL` | `runwayml/stable-diffusion-inpainting` | HuggingFace model ID for inpainting |

### Setting Environment Variables

**Windows PowerShell:**
```powershell
$env:SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
$env:INPAINT_MODEL   = "runwayml/stable-diffusion-inpainting"
$env:CORS_ORIGINS    = "http://localhost:3000"
```

**macOS / Linux:**
```bash
export SEGFORMER_MODEL="nvidia/segformer-b0-finetuned-ade-512-512"
export INPAINT_MODEL="runwayml/stable-diffusion-inpainting"
export CORS_ORIGINS="http://localhost:3000"
```

**Using a `.env` file** (recommended):

Create `backend/.env`:
```
SEGFORMER_MODEL=nvidia/segformer-b0-finetuned-ade-512-512
INPAINT_MODEL=runwayml/stable-diffusion-inpainting
CORS_ORIGINS=http://localhost:3000
```

### Low-VRAM Configuration

If you have less than 6 GB of GPU VRAM, add the following to `backend/app/ai/loaders.py` inside `get_inpaint_pipeline()`:

```python
pipe.enable_model_cpu_offload()   # offloads unused layers to CPU RAM
pipe.enable_attention_slicing(1)  # reduces peak VRAM per attention step
```

---

## 9. Hardware Requirements

| Setup | GPU VRAM | Est. Time per Image | Notes |
|---|---|---|---|
| CPU only (no GPU) | — | 10–20 minutes | Works, but patience required |
| Budget GPU (GTX 1060 6GB) | 6 GB | 45–90 seconds | Use `enable_model_cpu_offload()` |
| Mid-range (RTX 3070 8GB) | 8 GB | 20–40 seconds | Comfortable |
| High-end (RTX 3090 24GB) | 24 GB | 10–20 seconds | Fast; can increase resolution |
| Professional (A100) | 40 GB+ | 5–10 seconds | Maximum quality |

**Disk space required:**
- Model cache: ~4.5 GB (downloaded once, stored in `~/.cache/huggingface/`)
- Outputs: ~2–5 MB per session

**RAM:**
- Minimum: 8 GB system RAM
- Recommended: 16 GB (to allow model CPU offloading without swapping)

---

## 10. Contributing

We welcome contributions of all kinds — bug fixes, new styles, UI improvements, documentation, and more.

### 10.1 Reporting Bugs

Please open a GitHub Issue with:
- A clear title and description of the problem
- Your OS, Python version, and GPU (if applicable)
- Steps to reproduce the issue
- The full error message or traceback if one exists

### 10.2 Suggesting Features

Open a GitHub Issue with the label `enhancement`. Describe what you'd like to see and why it would be useful.

### 10.3 Submitting Code Changes

1. **Fork** the repository on GitHub
2. **Create a branch** for your change:
   ```bash
   git checkout -b feature/add-new-style
   ```
3. **Make your changes** (see Development Setup below)
4. **Test your changes** manually using the API docs at `/docs`
5. **Commit** with a clear message:
   ```bash
   git commit -m "Add Coastal Breeze style with ocean-inspired palette"
   ```
6. **Push** to your fork and open a **Pull Request**

### 10.4 Development Setup

```bash
# Install development extras
pip install -r requirements.txt
pip install black ruff

# Format code before committing
black backend/
ruff check backend/

# Verify no import errors
python -m compileall backend/app
```

### 10.5 Adding a New Design Style

To add a new style (e.g., "Industrial Loft"):

1. Open `backend/app/ai/styles_config.py`
2. Add a new entry to the `STYLE_CONFIG` dictionary:

```python
"industrial_loft": {
    "label": "Industrial Loft",
    "prompt": (
        "photorealistic industrial kitchen with exposed concrete surfaces, "
        "brushed steel countertops, raw concrete floor, brick backsplash, "
        "urban loft aesthetic, moody lighting, interior photography"
    ),
    "negative_prompt": (
        "marble, wood, white, bright, cartoon, blurry, low quality, watermark"
    ),
    "strength": 0.80,
    "guidance_scale": 8.5,
    "num_inference_steps": 30,
},
```

3. Add the style to the frontend style selector button list
4. Test via `curl http://127.0.0.1:8001/api/styles` and confirm your new style appears

### 10.6 Coding Standards

- **Python style**: Follow [PEP 8](https://peps.python.org/pep-0008/). Use `black` for formatting.
- **Type hints**: Add type hints to all function signatures
- **Docstrings**: Add a one-line docstring to every function
- **No hardcoded paths**: Use `pathlib.Path` and `config.py` settings; never hardcode filesystem paths
- **Error handling**: All API endpoints should return appropriate HTTP status codes with descriptive messages

---

## 11. Roadmap & Future Enhancements

### Near-term (MVP Completion)

- [ ] Add mask editing UI — let users manually correct segmentation mistakes using a brush tool
- [ ] Add loading progress indicator to the frontend during generation
- [ ] Implement async job queue (Celery + Redis) so generation runs in the background
- [ ] Add a segmentation confidence visualiser showing how certain the model is about each region

### Medium-term

- [ ] **More styles** — Industrial Loft, Coastal Breeze, Japandi Minimal, Art Deco Gold
- [ ] **Partial application** — clickable style per region (marble floor + wood cabinets)
- [ ] **Shareable results** — generate a shareable link with before/after images
- [ ] **Higher quality** — upgrade to SDXL or Flux Fill for 1024×1024 output
