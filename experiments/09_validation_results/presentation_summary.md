# Validation Search Results

- Generated top-k candidate results for the validation query set.
- Stored separate CSV outputs for FAISS, Qdrant, filtering, and reranking systems.
- Preserved a combined CSV for manual relevance labeling.
- Logged skipped systems and placeholder image-to-image queries in `run_summary.md`.

| System | Purpose |
| --- | --- |
| FAISS baseline | Exact CLIP semantic baseline |
| FAISS + style rerank | Tests whether style-aware reranking improves visual style matching |
| Qdrant semantic | Production-like vector database search |
| Qdrant filtered | Tests payload filters for style/object/keyword control |
| Qdrant object rerank | Tests object-aware reranking over semantic candidates |

This stage connects the prepared validation set to the final evaluation pipeline by producing ranked candidates ready for manual relevance labels.
