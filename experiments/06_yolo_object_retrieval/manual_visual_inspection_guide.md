# Manual Visual Inspection Guide

Use `visual_inspection.csv` to label the top-10 grids from the YOLO object retrieval experiment.

## Label Scale

- `2` = full visual match
- `1` = partial visual match
- `0` = visual failure

## Object Precision@10 (required)

For every result in the `qdrant_semantic`, `qdrant_object`, and `qdrant_object_rerank` rows, fill `object_present` with exactly `1` when the requested object is visibly present and `0` when it is absent. Do not infer this from the YOLO payload or metadata; inspect the image. Leave no blank cells in these three modes for the 12 object queries if you want the aggregate metric.

The aggregate `object_precision_metrics.csv` reports Object Precision@10 before (`qdrant_semantic`) and after object filter/rerank (`qdrant_object`, `qdrant_object_rerank`).

## Object-Like Queries

- `2` = requested object is clearly visible and central
- `1` = requested object is present but small, ambiguous, or not central
- `0` = requested object is not visible

## Combined Queries

- `2` = requested object and requested context are both visible
- `1` = only object or only context is visible
- `0` = neither is visible

## Failure Reasons

Use one of these values in `failure_reason`:

- `good_match`
- `partial_match`
- `wrong_object`
- `wrong_scene`
- `object_too_small`
- `style_or_context_mismatch`
- `yolo_miss`
- `metadata_noise`
- `unsupported_class`
- `too_generic`

Leave fields empty until you have inspected the corresponding image.
