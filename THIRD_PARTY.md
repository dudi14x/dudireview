# Model provenance

ArchVision: https://github.com/Parth-Talsania/Archvision

Inspected commit: `5b1b4a307e43c94164b7b859e7fb2f5e656c115f`.
Checkpoint: upstream `models/best.pt`, retained locally as `models/archvision-rooms.pt`.
SHA-256: `639c5deab18bcde81104f958f991175a609d0ebe909970ee5265bc7370b23b17`.
The upstream MIT notice is preserved at `third_party/ARCHVISION_LICENSE`.
The original adapter uses the checkpoint for polygon predictions and adds tiled inference, duplicate suppression, metric geometry and review.
Upstream reports training YOLOv8s-seg on 200 augmented images and 12 validation images. Its reported metrics are not independently reproduced here and are not accuracy estimates for tower/engineering plans.
Ultralytics has its own AGPL-3.0 / Enterprise licensing terms; the upstream MIT notice does not replace dependency or training-data licenses.

The user-uploaded FloorReview_Web_Review.zip was inspected as a reference. Its vector measurement helper is retained as `floorreview/legacy_measure_pdf.py`; the dashboard and analysis backend are newly implemented. No large uploaded pickle or PyTorch checkpoint has been deserialized.
