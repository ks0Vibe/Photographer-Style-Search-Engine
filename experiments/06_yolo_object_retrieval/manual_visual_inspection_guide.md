# Manual Visual Inspection Guide

Use `visual_inspection.csv` to label the top-10 grids from the YOLO object retrieval experiment.

## Label Scale

- `2` = full visual match
- `1` = partial visual match
- `0` = visual failure

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
