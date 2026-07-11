# Relevance Labeling Guide

Human labels should be based primarily on visual relevance to the query. Do not judge a result as relevant only because the CLIP score is high, keywords match, or YOLO detected an object.

## What to Inspect

Open or preview the image referenced by `image_path`, then compare the visible content to the `query` and `type` fields. Use metadata fields only as supporting context.

## Relevance Scale

| Label | Meaning |
| ---: | --- |
| 2 | Highly relevant |
| 1 | Partially relevant |
| 0 | Not relevant |

### 2 = Highly Relevant

The result strongly satisfies the query.

- Correct main object.
- Correct visual scene.
- Correct requested photographic style.
- Correct object and style combination for mixed queries.
- Strong visual similarity for image-to-image queries.

### 1 = Partially Relevant

The result satisfies part of the query but misses an important element.

- Correct object but wrong scene.
- Correct style but weak semantic match.
- Correct scene but missing requested object.
- Visually related but not a strong match.

### 0 = Not Relevant

The result does not meaningfully satisfy the query.

- Wrong object.
- Unrelated scene.
- Opposite visual style.
- Accidental keyword or detector match.
- Visually unrelated image.

## Query-Type Guidance

### Semantic Queries

Judge whether the main scene or subject visually matches the query. Style can help, but semantic content should drive the label.

### Style Queries

Inspect mood, lighting, color, contrast, saturation, and composition. The exact object may be less important unless the query names one.

### Object Queries

Inspect whether the requested object is actually visible. Do not trust YOLO blindly; detector labels are hints, not evidence.

### Mixed Queries

Require both semantic or object relevance and the requested style or scene. A result that satisfies only one side is usually `1`, not `2`.

### Image-to-Image Queries

Judge visual similarity to the query image, including subject, scene, composition, color, lighting, and photographic mood.

## Confidence

Confidence is optional and may remain empty during initial labeling.

- `high`: the label is obvious.
- `medium`: some ambiguity exists.
- `low`: the query or image is difficult to judge.

## Ambiguous Cases

Use `1` when a result is plausibly related but incomplete. Use `confidence=low` and add a short comment when the image is hard to interpret, the query is broad, or metadata conflicts with the visible image.
