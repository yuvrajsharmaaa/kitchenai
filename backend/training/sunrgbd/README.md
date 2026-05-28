# SUN RGB-D SegFormer Pipeline

This folder converts original SUN RGB-D annotations into semantic segmentation
masks for HuggingFace Transformers and SegFormer training.

Target classes:

- 0 background
- 1 floor
- 2 wall
- 3 cabinet
- 4 countertop
- 5 furniture
- 6 appliances

## 1) Convert SUN RGB-D

Your extracted SUN RGB-D folder is:

```powershell
C:\Users\Asus\Desktop\kitchyen\SUNRGB\SUNRGBD\SUNRGBD
```

Run the converter from this folder:

```powershell
cd C:\Users\Asus\Desktop\kitchyen\backend\training\sunrgbd
python convert_sunrgbd.py --output-dir dataset
```

For a quick smoke test:

```powershell
python convert_sunrgbd.py --output-dir dataset_test --limit 20
```

The converter reads `annotation2Dfinal/index.json` first, then falls back to
`annotation2D3D/index.json` and `annotation/index.json`. It rasterizes object
polygons with OpenCV and saves masks aligned to the copied RGB images.

Output:

```text
dataset/
  images/
  masks/
  metadata/
    dataset_info.json
    label_map.json
    samples.csv
    samples.jsonl
```

## 2) Create Train/Validation Splits

```powershell
python split_dataset.py --dataset-dir dataset --val-split 0.1 --seed 42
```

This writes:

```text
dataset/metadata/train.csv
dataset/metadata/val.csv
dataset/metadata/train.jsonl
dataset/metadata/val.jsonl
dataset/metadata/train.txt
dataset/metadata/val.txt
```

## 3) Visualize Masks

```powershell
python visualize_masks.py --dataset-dir dataset --count 5
```

Each visualization contains the RGB image, colorized mask, and overlay side by
side.

## 4) Train SegFormer

```powershell
python train_segformer.py --data-dir dataset --output-dir outputs/segformer_sunrgbd
```

`train_segformer.py` reads `metadata/train.csv` and `metadata/val.csv` from the
flat dataset layout. It also still supports the older layout:

```text
data/
  train/images/
  train/masks/
  val/images/
  val/masks/
```

## Files

- `convert_sunrgbd.py`: full SUN RGB-D preprocessing script.
- `label_mapping.py`: class IDs, colors, and SUN RGB-D label aliases.
- `split_dataset.py`: reproducible train/validation split generator.
- `visualize_masks.py`: mask colorization and overlay preview script.
- `configs/label_map.json`: class ID mapping for training/config use.
