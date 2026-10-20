# FloorReview

Upload architectural drawings and receive a dashboard of proposed floor geometry, room schedules, candidate apartment mix, occupancy and MEP spaces. All metric areas are square metres.

## Run

Python 3.12, Git, and CPU are sufficient; CUDA is optional for model training.

```bash
bash scripts/setup.sh
bash scripts/start.sh
```

The server listens on port 8765, loopback by default. Use your hosting platform's authenticated forwarding for access. `PORT`, `FLOORREVIEW_HOST`, `FLOORREVIEW_DATA`, and `FLOORREVIEW_MODEL` are optional overrides. This is a local/single-workspace application, not a multi-tenant authenticated public service. Do not expose an unauthenticated public deployment.

## Dashboard workflow

1. Select one or multiple PDF/PNG/JPEG files. The durable queue admits up to 1,000 pending drawings, two processing workers, 32 MiB per upload. Select a PDF page (one selected page per document per submission). CAD formats require PDF export.
2. Open a finished result. View room polygon proposals over the plan, candidate 1/2/3-bed mix, MEP count, room schedule and occupancy totals.
3. Calibrate using two image points and a known real distance in metres. A unique printed PDF `1:N` ratio supplies an explicitly unverified scale estimate; raster images require calibration. Do not infer scale from a page's pixel width alone.
4. Review the proposed floor outline and room polygons. Trace/edit them when needed. Assign stairs, corridors, walls and voids as excluded regions. Assign residential / MEP / commercial / common occupancy and apartment unit IDs. Candidate unit markers do not establish unit boundaries or assign rooms to apartments automatically.
5. Correct the unit register, mark review complete, save, and download JSON/CSV or reviewed training labels.

### Area definitions

- Gross floor area: traced outer footprint polygon, including walls and circulation. This is a geometric definition, not a jurisdiction-specific regulatory GFA calculation.
- Net room area: union of room polygons clipped to the footprint, minus explicitly classified exclusions. Overlaps are counted once in the total.
- Individual room areas: each room minus exclusions. Overlapping room entries are flagged because individual sums can exceed the union.
- Unallocated area: footprint minus occupied room union and exclusions. Unknown residual is never automatically called a wall.
- Group totals: polygon unions within occupancy and unit assignments. Different overlapping groups can still overlap and are flagged.

## AI architecture

`upload → persistent SQLite queue → PDF rasterization/image decoding → tiled pretrained YOLO room segmentation → polygon cleanup → optional embedded-PDF label evidence → metric geometry → reviewable dashboard → export/training labels`.

ArchVision's checkpoint is the default room detector. It learns image geometry; areas come from calibrated polygons, not printed area text. Optional embedded PDF labels supplement naming/classification and unit-marker candidates. Scanned-plan text OCR is not implemented. No label means unclassified, not an invented occupancy. MEP and apartment grouping across arbitrary plan styles remain supervised review tasks until representative models and validation exist.

The fallback geometry detector and CPU random forest are available by setting `FLOORREVIEW_MODEL` to a missing file or `models/segmenter.joblib`. `models/plan-unet.pt` selects the compact neural segmentation model. Training does not silently replace the active detector.

## Training

See [docs/TRAINING.md](docs/TRAINING.md). Both a CPU random forest and compact U-Net have been trained on generated line drawings. This checks the training/inference machinery, not accuracy on real plans; synthetic diagrams are deliberately easy and lack stairs/corridor/MEP labels. No new model has been trained on the private uploaded PDFs without ground-truth labels.

Use reviewed exports or licensed datasets to produce masks; split by building/project before augmentation and retain an untouched test set. The provided larger CubiCasa archive and checkpoints exceed the upload download tool's 32 MiB transfer cap and were not loaded.

## Validation

```bash
.venv/bin/python -m pytest -q
node --check floorreview/static/app.js
.venv/bin/python -m pip check
```

Backend tests exercise area arithmetic, overlapping rooms, exclusions, geometry rejection, scale absence, PDF/image upload, review persistence, stale updates, results and mask exports, actual training/inference, queue success/failure and 1,000-item admission capacity. Capacity testing is not a benchmark of 1,000 full model inferences. Browser checks use Chromium; real uploaded PDFs are exercised separately without claiming ground-truth accuracy.

Model weights, private drawings, training datasets, output results and the virtual environment are ignored by Git. `scripts/fetch_model.py` reproduces the exact upstream checkpoint with checksum verification. `scripts/start.sh` restarts the service; live processes are not part of a saved environment snapshot.
