# Release v0.1 — KitchenAI MVP

This release contains the initial KitchenAI MVP:

- FastAPI backend for image upload, segmentation, and inpainting
- SegFormer-based segmentation pipeline and utilities
- Diffusers ControlNet inpainting pipeline integration
- SUN RGB-D preprocessing and dataset conversion scripts (training helpers)
- Simple browser UI served by the backend for quick testing
- CI/ignore rules to avoid committing large datasets and model artifacts

Notes:
- Do not push `backend/outputs/`, model checkpoints, or `SUNRGB/` dataset files.
- For full end-to-end inpainting you may need to download large model weights locally.

How to create the PR: this branch was pushed as `release/v0.1` and a pull request can be opened against `main`.
