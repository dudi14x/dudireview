# Data and model plan for reliable plan analysis

The existing ArchVision detector is a room-instance proposal baseline. It is not a validated wall tracer, complete apartment detector or MEP classifier. Synthetic model training verifies plumbing; it does not establish accuracy on engineering plans.

## Suitable sources

| Source | Useful supervision | Limits for this workflow |
| --- | --- | --- |
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) and its [dataset download](https://zenodo.org/record/2613548) | 5,000 floor-plan samples; polygon annotations over 80 object categories; room/wall/junction learning | Map its taxonomy to our targets. Do not assume it labels MEP departments or apartment memberships in tower projects. Verify dataset terms before intended use. Its original code uses an old PyTorch stack and needs an isolated integration or modernization. |
| [FloorPlanCAD](https://floorplancad.github.io/) | Vector CAD line/symbol understanding | Supporting task for CAD exports; not a substitute for complete room/unit/MEP labels. Verify access, annotation categories and license. |
| [RoomFormer](https://github.com/ywyue/RoomFormer) with [Structured3D](https://github.com/bertjiazheng/Structured3D) | Structured prediction of closed room polygons | Official pretrained inputs are point-cloud-derived top-down density maps, not arbitrary rasterized architectural PDFs. Directly loading these weights is not a validated PDF solution. Structured3D access is subject to its agreement. |
| Representative user projects | MEP service rooms/shafts, apartment membership, bedrooms per unit, stairs/circulation, exterior boundaries and true scale | Requires reviewed annotations; do not turn current wrong detections into training ground truth. |

Kaggle is useful for GPU training and hosting a properly licensed dataset. A mirror does not establish provenance, license or class coverage. No Kaggle key is needed merely to plan this work.

## Training sequence

1. Start from a room/wall segmentation model and representative public labels, not the synthetic demo alone. Preserve closed boundaries through polygonization and topology checks.
2. Fine-tune on corrected examples from the actual input styles: colored concept sheets, detailed construction PDFs, scans and multi-wing layouts. Include hard negatives such as dimension boxes, key plans, legends and courtyards.
3. Train occupancy/MEP classification and apartment grouping with their own ground truth. A bedroom polygon is not an apartment; PDF markers are candidates, not unit boundaries.
4. Use page scale/reference geometry separately from learned semantics. Area is calculated from calibrated geometry, not copied from printed schedule text.
5. Split by building/project before augmentation. Adjacent floors, revisions and crops of the same building must remain in one split. Keep a final untouched test set.
6. Measure wall/room IoU, room instance precision/recall, boundary and topology errors, absolute/relative gross and room area error, unit count/grouping errors, and MEP precision/recall. Report missing and unclassified spaces, not only successful detections.

Begin with a reviewed pilot set spanning the actual styles before deciding the full labeling volume. There is no justified accuracy percentage or sufficient sample-count guarantee yet. Dataset conversion, compatibility and model evaluation must be completed before promoting new weights. GPU fine-tuning of a large labeled real dataset has not been performed in this environment.

## Current corrective processing

A PDF's explicit color legend can recover vector MEP/exclusion regions and reject unsupported detections. It does not work on every scan or uncolored CAD page and is not a general learned classifier. Closed dark linework can propose an exterior boundary, but scale and boundary confirmation remain separate. Missing geometry cannot be repaired by changing scale.
