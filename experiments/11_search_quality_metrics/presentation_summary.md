# Search Quality Evaluation

- Evaluated 28 queries across 5 retrieval systems.
- Relevance scale: 2 highly relevant, 1 partially relevant, 0 not relevant.
- Binary metrics use relevance >= 1 as relevant.
- Best overall system by nDCG@10: `qdrant_filtered`.

| System | P@5 | P@10 | nDCG@10 | MRR | Success@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| qdrant_filtered | **0.964** | **0.946** | **0.948** | **0.982** | **1.000** |
| qdrant_object_rerank | 0.950 | **0.946** | 0.939 | **0.982** | **1.000** |
| faiss_style_rerank | 0.907 | 0.886 | 0.916 | 0.940 | **1.000** |
| faiss_baseline | 0.929 | 0.921 | 0.913 | 0.946 | **1.000** |
| qdrant_semantic | 0.929 | 0.921 | 0.913 | 0.946 | **1.000** |

## Best System by Query Group

- `mixed_text`: `qdrant_filtered` (nDCG@10=0.936)
- `object_text`: `qdrant_filtered` (nDCG@10=0.989)
- `semantic_text`: `faiss_style_rerank` (nDCG@10=0.942)
- `style_text`: `qdrant_filtered` (nDCG@10=0.948)

Shared human judgments make the system comparison fair because each query-image pair is labeled once and reused across retrieval methods.
