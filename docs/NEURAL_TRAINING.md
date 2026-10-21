# Train the floor-plan network

The app includes a six-class U-Net (background, walls, rooms, stairs, corridors,
MEP). It produces semantic masks; geometry turns connected regions into the
existing clickable dashboard phases. It does not yet learn apartment ownership
or bedroom counts. Those still use the unit register and available PDF markers.
Physical scale detection is separate from this network.

## Real drawing training

1. Correct the regions and outer boundary in the dashboard. Mark the boundary
   checked and the drawing reviewed, save, then Export training labels.
   Unknown interior pixels are ignored, not invented as background. Include
   correctly traced examples of all six classes, especially MEP and exclusions.
2. Put exports in a folder and create `manifest.json` beside them:

```json
[
  {"archive":"building-a-floor1.zip","building":"A","split":"train"},
  {"archive":"building-b-floor1.zip","building":"B","split":"val"},
  {"archive":"building-c-floor1.zip","building":"C","split":"test"}
]
```

All pages from one building must share a split. Duplicate drawings are rejected.
Validation selects the checkpoint; keep the test building for independent final
assessment. This example demonstrates the format, not a sufficient dataset size.

3. Windows: run `TRAIN.bat path\to\manifest.json` from the project folder.
   START.bat must have installed the environment first. The supplied Windows
   setup uses CPU PyTorch; GPU training requires a compatible CUDA installation.
   Python users can run:

```sh
python -m floorreview.dataset --manifest labels/manifest.json --output datasets/reviewed
python -m floorreview.neural --data datasets/reviewed --output models/reviewed-plan-unet.pt --epochs 60
```

To fine-tune an existing compatible checkpoint, add
`--resume models/plan-unet.pt`. Only use trusted local checkpoints.

4. Inspect the metrics and history JSON next to the checkpoint. Reported IoU is
   evaluated at 256×256, which can hide thin-line errors; it is not evidence of
   real-world area accuracy. Independently compare full-size predictions and
   measured areas on unseen buildings before accepting the model.
5. Add `FLOORREVIEW_MODEL=models/reviewed-plan-unet.pt` to `.env.local`, restart
   START.bat, then re-upload drawings. Activation is explicit; training never
   silently replaces the deployed model. Remove the setting to return to
   ArchVision. Untrained or absent classes cannot be considered reliable.

## Verify the pipeline without real labels

```sh
python -m floorreview.neural --data datasets/smoke --output models/smoke-plan-unet.pt --epochs 1 --synthetic
```

This trains on generated rectangles solely to check execution. It does not train
MEP, apartment grouping or a useful production model. No real building accuracy
claim follows from this smoke run.
