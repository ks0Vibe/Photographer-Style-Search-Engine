# Qdrant Index Selection Benchmark

This benchmark uses an exact FAISS `IndexFlatIP` over the same synthetic 500k vectors as ground truth. Every Qdrant variant has explicit HNSW settings and is queried with the same normalized query vectors.

- Corpus: `500000` vectors, 512 dimensions
- Exact FAISS ground-truth build time: `1.764` s
- Recall@10 is against the exact FAISS top-10 set; p50/p95 are Qdrant-only search latency after one warm-up query per query vector.
- `native_scalar_int8=true` means Qdrant scalar quantization is configured server-side; no Python dequantization is included.

## Results

| variant | collection_size | m | ef_construct | ef_search | native_scalar_int8 | build_time_seconds | recall_at_10 | p50_latency_ms | p95_latency_ms | disk_after_mb | disk_delta_mb | container_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hnsw_m16_efc100_efs64 | 500000 | 16 | 100 | 64 | False | 274.977 | 1.0 | 5.829 | 28.729 | 2700.723 | 2700.723 | 1405.952 |
| hnsw_m32_efc200_efs128 | 500000 | 32 | 200 | 128 | False | 316.359 | 1.0 | 7.121 | 30.529 | 2640.715 | 2640.715 | 1330.176 |
| hnsw_m64_efc400_efs256 | 500000 | 64 | 400 | 256 | False | 311.068 | 1.0 | 26.055 | 34.739 | 2645.52 | 2645.52 | 1277.952 |
| native_scalar_int8_m32_efc200_efs128 | 500000 | 32 | 200 | 128 | True | 279.352 | 0.8833 | 6.345 | 31.097 | 2948.755 | 2948.755 | 1666.048 |

## Interpretation rule

Select the lowest-latency configuration whose Recall@10 meets the project threshold (recommended >= 0.98). If native INT8 meets that threshold, it is the production-like memory winner; otherwise select the HNSW setting on the Pareto frontier and keep INT8 as a measured alternative.

## Reproduce

```powershell
$env:QDRANT_MODE='server'
$env:QDRANT_URL='http://localhost:6333'
.\.venv\Scripts\python.exe experiments\scripts\benchmark_qdrant_index.py
```
