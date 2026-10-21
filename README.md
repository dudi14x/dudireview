# FloorReview

Upload architectural drawings and receive a dashboard of proposed floor geometry, room schedules, candidate apartment mix, occupancy and MEP spaces. All metric areas are square metres.

## Run

Use 64-bit Python 3.10, 3.11 or 3.12. Python 3.12 is tested on Linux; Windows wheel availability is checked for Python 3.10. CUDA is optional for training. Git is not required to run a downloaded ZIP.

On Windows, extract the ZIP completely and double-click `START.bat`. The installer selects Python 3.12/3.11/3.10 through the Windows `py` launcher when available and uses a separate `.venv-win` folder. Other Python versions can remain installed. Keep the terminal open, then visit `http://localhost:8765`. If Python is missing, install Python 3.12 (64-bit) and select **Add python.exe to PATH**.

If an older download fails with `torch==2.14.1`, download the latest branch ZIP and rerun `START.bat`. The corrected installer uses the matching CPU pair `torch==2.8.0` / `torchvision==0.23.0`. The Windows installer uses `.venv-win` to avoid incompatible environments left by earlier installers; deleting your drawings or results is unnecessary.

```bash
bash scripts/setup.sh
bash scripts/start.sh
```

The server listens on port 8765, loopback by default. Use your hosting platform's authenticated forwarding for access. `PORT`, `FLOORREVIEW_HOST`, `FLOORREVIEW_DATA`, and `FLOORREVIEW_MODEL` are optional overrides. This is a local/single-workspace application, not a multi-tenant authenticated public service. Do not expose an unauthenticated public deployment.

## Dashboard workflow

1. Select one or multiple PDF/PNG/JPEG files. The durable queue admits up to 1,000 pending drawings, two processing workers, 32 MiB per upload. Select a PDF page (one selected page per document per submission). CAD formats require PDF export.
2. Open a finished result. View room polygon proposals over the plan, candidate 1/2/3-bed mix, MEP count, room schedule and occupancy totals.
3. Physical scale is detected automatically from matched dimension lines or a graphic scale bar with units. A printed ratio alone supplies an explicit estimate. Use **Measure distance** to select two points and receive their computed separation; no typed reference length is needed when scale detection succeeds. Raster-only images without a usable reference remain unresolved. Manual calibration is an optional advanced override.
4. Review the proposed floor outline and room polygons. The outline now comes from closed dark linework, never a connect-the-dots envelope of room detections. Check **Exterior boundary checked against the drawing**, or trace the perimeter yourself, before a gross floor measurement is reported. Trace/edit them when needed. Assign stairs, corridors, walls and voids as excluded regions. Assign residential / MEP / commercial / common occupancy and apartment unit IDs. Candidate unit markers do not establish unit boundaries or assign rooms to apartments automatically.
5. Correct the unit register, mark review complete, save, and download JSON/CSV or reviewed training labels.

### Area definitions

- Gross floor area: traced outer footprint polygon, including walls and circulation. This is a geometric definition, not a jurisdiction-specific regulatory GFA calculation.
- Net room area: union of room polygons clipped only to a user-confirmed footprint, minus explicitly classified exclusions. An unconfirmed outline never clips room areas. Overlaps are counted once in the total.
- Individual room areas: each room minus exclusions. Overlapping room entries are flagged because individual sums can exceed the union.
- Unallocated area: footprint minus occupied room union and exclusions. Unknown residual is never automatically called a wall.
- Group totals: polygon unions within occupancy and unit assignments. Different overlapping groups can still overlap and are flagged.

## AI architecture

`upload → persistent SQLite queue → PDF rasterization/image decoding → tiled pretrained YOLO room segmentation → polygon cleanup → optional embedded-PDF label evidence → metric geometry → reviewable dashboard → export/training labels`.

ArchVision's checkpoint is the default room detector. When a vector PDF contains an explicit color legend, matching vector fills provide MEP/circulation/exterior-zone evidence, and unsupported model proposals outside those filled spaces are filtered. Colors without a matching legend do not establish an occupancy. It learns image geometry; areas come from calibrated polygons, not printed area text. Optional embedded PDF labels supplement naming/classification and unit-marker candidates. Scanned-plan text OCR is not implemented. No label means unclassified, not an invented occupancy. MEP and apartment grouping across arbitrary plan styles remain supervised review tasks until representative models and validation exist.

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

If setup reports no compatible Python, install the **Windows installer (64-bit)** from https://www.python.org/downloads/release/python-31210/ and enable the launcher. Then run `START.bat` again. You do not need to uninstall newer Python versions. On Windows the environment is `.venv-win`; run tests with `.venv-win\Scripts\python.exe -m pytest -q`.

Accuracy and dataset planning: [docs/DATASET_PLAN.md](docs/DATASET_PLAN.md). Updates to geometry processing require re-uploading an old drawing to run detection again; opening old results only refreshes their measurement-status calculations.


## Three clickable phases

The result cards are keyboard-accessible tabs:

1. **Total floor area** shows automatic scale evidence, floor-area estimates and distance measurement tools. Room overlays are hidden to keep the boundary visible.
2. **Individual rooms and exclusions** opens the room schedule, boundaries, and stairs/corridor/wall/void roles.
3. **Occupancy, residential units and MEP** opens occupancy labels, unit register, MEP-only highlighting, and locate controls.

The trained local vision model runs after upload. Phase switching uses the saved analysis and does not re-upload or call an API. Gemini is an optional visual second opinion, not OCR transcription, model fine-tuning, or a replacement for geometric tracing.

## Optional Gemini visual review

1. Get a key at https://aistudio.google.com/apikey. Check Google's current availability, free-tier limits, pricing and data terms. This application cannot guarantee free API usage.
2. Copy `.env.example` to `.env.local` in the project folder. Windows START.bat creates a blank-key template if the file is missing. Open it in Notepad and set `GEMINI_API_KEY=your-key`. Do not paste the real key into chat, GitHub or browser code.
3. Restart START.bat. Environment variables override `.env.local`.
4. Upload/open a drawing, choose a phase, save any edits, and check the box authorizing that drawing image to be sent to Google. Click **Review this phase with Gemini**.
5. Review the returned issues and proposed labels. Select individual suggestions, then **Apply selected label suggestions**. Geometry, calibration, unit counts and apartment assignments are preserved. Applying a room role such as corridor can change exclusion-area calculations, so review those roles.

Default model: `gemini-2.5-flash`. Set `GEMINI_MODEL` to an image-capable model available to your Google project if needed. The default local daily cap is five attempted calls, counted by UTC date; `GEMINI_DAILY_REQUEST_LIMIT=0` disables new requests. Errors count as attempts and are never automatically retried. Limits are per application data directory, not a Google billing cap. Google can still charge according to your account settings.

Only an annotated image overview and up to eight detail crops leave the server; the original PDF, API key and printed area totals are not sent as text. Up to 60 detected region IDs are considered in one review. Responses are validated and cached by model, phase, image and drawing state. Cache hits do not consume the local cap. A changed drawing invalidates old suggestions. Local processing remains usable without a key, network access or quota.

The API key stays server-side; `.env.local` is ignored by Git. Do not share that file. To use Gemini in the cloud environment, enter the key through environment settings instead of a local file. Live Gemini behavior is unverified until a valid key is provided; automated tests simulate provider responses, missing keys, quota errors, caching, stale results and invalid model output. No real API request is made by the test suite.


## Automatic distances and scale

PDF uploads now inspect geometric references automatically. A graphic scale bar needs at least three consistent numeric tick labels and a nearby unit. Dimension-based detection needs at least three consistent long dimension-line matches and two distinct values, with explicit units/unit notes or a cross-check against a declared scale. Conflicting bars, multiple declared viewport scales and materially competing dimension groups are rejected. A printed scale ratio alone remains an estimate because resized sheets can invalidate it.

The physical scale is converted to the actual rendered image resolution, including rotated PDFs. Matched reference evidence and the method are stored in `scale_detection`. The original PDF is retained privately in the local data folder for rerunning **Detect scale automatically**. Older results without the original PDF must be re-uploaded to use this feature. Original PDFs are not sent to Gemini.

Detected room areas appear automatically when physical scale is resolved. A proposed exterior outline also yields a visibly approximate gross-floor estimate (`≈`); this does not certify boundary correctness or fill missed rooms. Checked measurements remain distinct in JSON/CSV, and room CSV now includes a separate `area_estimate_m2` column. Manual reference input is under **Advanced measurement override**, not the normal upload workflow.

No vision model can infer absolute metres reliably from a drawing that contains no physical scale information. In that case the app reports **Scale unavailable**, explains the missing evidence and leaves distance/area unresolved rather than inventing a length. This version does not perform OCR-based calibration of scanned raster pages. Automatic scale detection does not repair inaccurate room boundaries or establish complete floor coverage.

### Train a reviewed-drawing neural network

See [the complete training workflow](docs/NEURAL_TRAINING.md). Windows users can run `TRAIN.bat path\to\manifest.json` to build a building-separated dataset and train a six-class U-Net. Activate it explicitly with `FLOORREVIEW_MODEL` in `.env.local`. Apartment grouping still uses the unit register; this network predicts space masks.

### Whole-drawing Gemini analysis

After configuring the key, select the image-sharing checkbox and use **Analyze all three phases with Gemini**. The response covers perimeter issues, room/exclusion roles, and occupancy/MEP labels for up to 60 supplied regions. **Download AI analysis JSON** combines current geometric measurements, unit candidates, scale evidence and separate AI suggestions. This is a bounded visual request, not ingestion of every page of a PDF. Apply selected suggestions to update classifications. No live Gemini verification is claimed without a user key.
