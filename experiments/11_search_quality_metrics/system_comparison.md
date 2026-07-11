# System Comparison

This experiment compares retrieval systems using shared human relevance judgments.

## Evaluation Setup

- Evaluated queries: 28
- Evaluated systems: 5
- Relevance scale: 2 = highly relevant, 1 = partially relevant, 0 = not relevant
- Binary relevance threshold: relevance >= 1
- Evaluation mode: complete/strict

## Main Comparison

| System | P@5 | P@10 | nDCG@10 | MRR | Success@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| qdrant_filtered | **0.964** | **0.946** | **0.948** | **0.982** | **1.000** |
| qdrant_object_rerank | 0.950 | **0.946** | 0.939 | **0.982** | **1.000** |
| faiss_style_rerank | 0.907 | 0.886 | 0.916 | 0.940 | **1.000** |
| faiss_baseline | 0.929 | 0.921 | 0.913 | 0.946 | **1.000** |
| qdrant_semantic | 0.929 | 0.921 | 0.913 | 0.946 | **1.000** |

## Best System by Metric

- `precision_at_5`: `qdrant_filtered`
- `precision_at_10`: `tie: qdrant_filtered, qdrant_object_rerank`
- `ndcg_at_10`: `qdrant_filtered`
- `mrr`: `tie: qdrant_filtered, qdrant_object_rerank`
- `success_at_5`: `tie: qdrant_filtered, qdrant_object_rerank, faiss_style_rerank, faiss_baseline, qdrant_semantic`

## Best System by Query Type

- `mixed_text`: `qdrant_filtered` (nDCG@10=0.936)
- `object_text`: `qdrant_filtered` (nDCG@10=0.989)
- `semantic_text`: `faiss_style_rerank` (nDCG@10=0.942)
- `style_text`: `qdrant_filtered` (nDCG@10=0.948)

## Quality Trade-Offs

Metric differences should be interpreted with the labeling status and query count in mind.
When systems tie or differ only by very small margins, the result should be treated as inconclusive until labels are complete.

## Limitations

- None
