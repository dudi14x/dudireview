# Validation performed

## Working application

- 12 pytest tests passed: metric geometry, overlapping rooms, exclusions, invalid geometry, absent calibration, actual PDF/image uploads, persistence, stale revision rejection, result/mask exports, random-forest training/inference, neural training/checkpoint loading, queue processing/failure/capacity, frame rejection and unit-marker convention/legend handling.
- `node --check floorreview/static/app.js` passed.
- `pip check` passed.
- Pinned setup script executed successfully, including exact upstream checkpoint SHA-256 verification.
- Chromium desktop and 390-pixel mobile checks: no JavaScript errors, upload through the dashboard, durable queued processing, view results, save a space and download exports. Mobile document width equals the viewport width.

The queue's 1,000-item admission limit was tested. A 1,000-drawing end-to-end model load/performance benchmark was not run.

## Actual supplied PDFs

Both supplied PDFs were processed using the pretrained ArchVision model with high-resolution overlapping tiles, partial-tile rejection and duplicate suppression. The final tested detector proposed 34 regions on the colored tower concept sheet and 116 on the first-floor engineering sheet. Those are candidates, not verified room counts. The pipeline also found candidate unit markers:

| Sheet | 1 bed | 2 bed | 3 bed |
| --- | ---: | ---: | ---: |
| Tower concept | 6 | 4 | 2 |
| First-floor engineering | 0 | 17 | 7 |

These counts come from PDF marker evidence, with legend entries and alternate BR/BD conventions filtered. They do not establish apartment boundaries or learn unit membership. Neither sample produced an MEP detection; this is a model/classification limitation, not proof that there are no MEP spaces. Room coverage, gross area, net area and unit counts have no reviewed ground truth here.

The concept sheet has no accepted scale and its metric fields remain unavailable until calibrated. The engineering PDF provides a unique printed 1:250 ratio, used as an unverified estimate. Automatic floor envelopes can include erroneous boundaries or omit external walls/wings and require review. Net totals refer to detected/edited room polygons, not guaranteed complete room coverage.

## Training performed

- CPU random forest: 48 generated train drawings and 12 generated validation drawings.
- Compact U-Net: 96 generated train drawings, 12 generated validation drawings, 8 epochs, saved best epoch 6.
- Generated examples contain only background/wall/room classes. Stairs/corridor/MEP have no training examples or validation score in those runs.
- Easy synthetic diagrams do not measure architectural-plan generalization. No newly trained model is claimed accurate on real drawings. The active model remains the pretrained ArchVision checkpoint.

A review-labeled, project-separated real dataset and held-out evaluation are required for the unattended three-phase target. A saved environment configuration is not publication or fresh-task restoration validation.
