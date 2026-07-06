# Validation Set for Search Quality

- Added a reusable 30-query validation set for final retrieval-quality evaluation.
- Covered semantic, style, object, mixed, and image-to-image search scenarios.
- Included a manual top-10 relevance-label template for future metric computation.
- Kept this stage separate from generated experiment outputs and synthetic-scale testing.

| Query group | Count |
| --- | ---: |
| semantic_text | 8 |
| style_text | 8 |
| object_text | 8 |
| mixed_text | 4 |
| image_to_image | 2 |
| Total | 30 |

Relevance will be labeled manually on a 0-2 scale: 2 for highly relevant, 1 for partially relevant, and 0 for not relevant.

This stage gives the final presentation a clear bridge from system implementation to measurable search-quality evaluation.
