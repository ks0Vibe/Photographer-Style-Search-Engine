# Validation Results

This experiment stage generates ranked top-k search outputs for the validation queries defined in `experiments/08_validation_set/queries.csv`.

The outputs are intended for the next evaluation step: manual relevance labeling and later computation of Precision@5, Precision@10, nDCG@10, and MRR. This stage does not compute those metrics.

## Run

```powershell
python experiments/scripts/generate_validation_results.py
```

Optional arguments:

```powershell
python experiments/scripts/generate_validation_results.py --queries-path experiments/08_validation_set/queries.csv --output-dir experiments/09_validation_results --top-k 10 --candidate-pool-size 100 --systems all
```

`--systems` accepts `all` or a comma-separated list of `faiss_baseline`, `faiss_style_rerank`, `qdrant_semantic`, `qdrant_filtered`, and `qdrant_object_rerank`.

## Outputs

| File | Meaning |
| --- | --- |
| `results_faiss_baseline.csv` | Exact FAISS CLIP semantic baseline results. |
| `results_faiss_style_rerank.csv` | FAISS semantic candidates reranked with the existing style reranker when style cues are available. |
| `results_qdrant_semantic.csv` | Qdrant semantic vector search without payload filters. |
| `results_qdrant_filtered.csv` | Qdrant results using inferred object/style payload filters when reliable, otherwise semantic fallback. |
| `results_qdrant_object_rerank.csv` | Qdrant semantic candidates reranked with object-aware reranking when an object can be inferred. |
| `all_validation_results.csv` | Concatenated rows from all generated system result CSVs. |
| `run_summary.md` | Run timestamp, completed/skipped systems, row counts, warnings, and skipped image-to-image queries. |

All result CSVs use the same schema so they can be copied into the manual relevance-labeling workflow.
