# Manual Relevance Assessment

- Deduplicated query-image pairs are labeled once, even when returned by multiple systems.
- Labels use a graded relevance scale: 0, 1, and 2.
- Shared labels are reused for fair comparison across retrieval systems.
- Progress tracking separates labeled and unlabeled judgments.

| Label | Meaning |
| ---: | --- |
| 2 | Highly relevant |
| 1 | Partially relevant |
| 0 | Not relevant |

A shared judgment pool ensures that all retrieval systems are evaluated against the same human relevance criteria.
