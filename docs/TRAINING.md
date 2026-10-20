# Training for varied drawings

The production goal is a model that generalizes across projects and drawing styles, not one that remembers a pair of sheets. Upload count and labeled training count are different.

## Annotation contract

Export reviewed labels from the dashboard. Each archive contains a rendered drawing, a single-channel semantic PNG mask, and polygon/unit metadata. Collect images and masks into matching file names:

```
dataset/
  train/images/*.png
  train/masks/*.png
  val/images/*.png
  val/masks/*.png
```

Mask IDs: `0 background`, `1 wall`, `2 room`, `3 stairs`, `4 corridor`, `5 MEP`, `255 unknown/ignored`. Unannotated space inside the footprint is ignored, never silently converted into a wall. Exported manual exclusions override room labels. Occupancy metadata and apartment membership remain in the JSON; the six-class pixel model does not learn apartment identifiers or bedroom counts.

Separate drawings by building/project, including adjacent floors and revisions. Do this before augmentation. Near-identical revisions in train and validation cause leakage; the scripts reject exact duplicate drawing pixels, but human grouping is still needed. Target a representative labeled set across scans, colored brochures, detailed CAD exports and project types, plus untouched real test drawings. More than 1,000 unreviewed uploads does not supply missing labels.

## Commands

Verify pipeline using simple generated diagrams:

```bash
.venv/bin/python -m floorreview.training --data data/synthetic --synthetic --output models/segmenter.joblib
.venv/bin/python -m floorreview.neural --data data/neural-synthetic --synthetic --epochs 8 --output models/plan-unet.pt
```

Train a neural model on annotated real drawings:

```bash
.venv/bin/python -m floorreview.neural --data dataset --epochs 60 --output models/reviewed-plan-unet.pt
```

The compact U-Net uses image lines and textures, random crops/flips, class-weighted pixel loss and unknown-pixel masking. It records per-class IoU, confusion matrix, validation count, training history and best-epoch checkpoint. Validation uses 256×256 images and is not a full-resolution polygon/area benchmark. The trainer streams one image/mask pair at a time instead of holding 1,000 decoded drawings in memory. A GPU is recommended for large real training sets. CPU training is verified for the small synthetic set.

Select a reviewed checkpoint explicitly with `FLOORREVIEW_MODEL=/absolute/path/to/models/reviewed-plan-unet.pt` and restart. The backend uses overlapping 256-pixel tiles for inference. Keep a previous checkpoint to roll back. Only locally controlled artifacts are loaded. Uploaded pickle/PT/PTH files are not executed.

## Transfer learning from ArchVision

The default ArchVision model already has room instance segmentation training. Fine-tuning should use YOLO segmentation polygons, not semantic masks copied without conversion. The adapter records source and detection confidence. The dashboard exports polygons as JSON; annotate and convert an approved dataset to YOLO's normalized `class x1 y1 ...` format, preserving train/validation project separation before running Ultralytics training. MEP and unit segmentation need dedicated classes/labels and evaluation beyond upstream's single room detector.

## Datasets

- CubiCasa5K: https://github.com/CubiCasa/CubiCasa5k — useful room/wall and icon supervision. Respect its license and class definitions. The repository source and model weights alone are not necessarily the complete labeled data.
- Structured3D: https://structured3d-dataset.org — useful structural layout supervision; check access and task fit.
- ArchVision's README references a Roboflow floor-plan segmentation dataset; check the exact dataset version, annotation format, license and permitted redistribution before using it.
- Kaggle can host a GPU training job or a licensed dataset mirror. Confirm source, license and wall/room/stairs/corridor/MEP coverage. No Kaggle credentials are required for the current application. Do not put API keys in source or chat.

## Acceptance before automatic area reporting

Measure held-out wall/room IoU, room instance precision/recall, floor-boundary error, calibrated gross/net area error, apartment-count error, apartment-membership accuracy and MEP precision/recall. Include failed pages and unknown classifications. The current two private PDFs have no reviewed polygon ground truth, so their processing counts are not accuracy measurements. Domain-specific retraining and independent scale verification remain necessary for reliable unattended results.

For a reviewed, local YOLO segmentation dataset:

```bash
.venv/bin/python scripts/train_rooms.py --data dataset/data.yaml --epochs 60 --batch 4 --device 0
```

Use `--device cpu` for CPU training. Check mask precision/recall and mAP on a separate project-level test split before selecting the checkpoint for inference. To use a fine-tuned YOLO checkpoint, copy it to a controlled `models/archvision-reviewed.pt` path and set `FLOORREVIEW_MODEL` to that path. Keep `unet` in the file name for compact U-Net checkpoints; other current PT paths should retain the `archvision` prefix for the YOLO adapter.
