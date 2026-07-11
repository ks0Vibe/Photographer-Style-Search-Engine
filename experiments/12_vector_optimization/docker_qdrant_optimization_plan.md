# Docker Qdrant Vector Optimization Plan

Use this plan to measure container-level memory and latency for the same 500k synthetic vector corpus under different Qdrant vector-storage settings.

## Baseline Float32 Collection

```powershell
docker compose up -d qdrant
docker stats --no-stream photographer-style-qdrant

$env:QDRANT_MODE="server"
$env:QDRANT_URL="http://localhost:6333"
$env:QDRANT_COLLECTION="photos_synthetic_500k_fp32"

.\.venv\Scripts\python.exe scripts\setup_optimized_qdrant_collection.py --datatype float32
.\.venv\Scripts\python.exe scripts\upload_synthetic_vectors_to_qdrant.py --batch-size 2048 --stats-path experiments\12_vector_optimization\synthetic_upload_fp32.json
.\.venv\Scripts\python.exe experiments\scripts\evaluate_synthetic_500k_scale.py --latency-runs 5
docker stats --no-stream photographer-style-qdrant
```

Save the generated latency summary before running the next variant, because `evaluate_synthetic_500k_scale.py` writes to the stage 07 output paths by default.

## Native Float16 Collection

```powershell
$env:QDRANT_COLLECTION="photos_synthetic_500k_float16"

.\.venv\Scripts\python.exe scripts\setup_optimized_qdrant_collection.py --datatype float16
.\.venv\Scripts\python.exe scripts\upload_synthetic_vectors_to_qdrant.py --batch-size 2048 --stats-path experiments\12_vector_optimization\synthetic_upload_float16.json
.\.venv\Scripts\python.exe experiments\scripts\evaluate_synthetic_500k_scale.py --latency-runs 5
docker stats --no-stream photographer-style-qdrant
```

Expected vector-memory effect: approximately 2x less vector storage than float32.

## Scalar Int8 Quantized Collection

```powershell
$env:QDRANT_COLLECTION="photos_synthetic_500k_scalar_int8"

.\.venv\Scripts\python.exe scripts\setup_optimized_qdrant_collection.py --datatype float32 --scalar-int8 --quantile 0.99 --on-disk
.\.venv\Scripts\python.exe scripts\upload_synthetic_vectors_to_qdrant.py --batch-size 2048 --stats-path experiments\12_vector_optimization\synthetic_upload_scalar_int8.json
.\.venv\Scripts\python.exe experiments\scripts\evaluate_synthetic_500k_scale.py --latency-runs 5
docker stats --no-stream photographer-style-qdrant
```

Expected vector-memory effect: quantized search representation is about 4x smaller than float32 vectors. Full container memory also includes HNSW graph, payloads, metadata, WAL, and allocator overhead.

## What To Put On The Slide

- Baseline vector memory estimate for 500k 512d float32: about 976.56 MB.
- Native float16 estimate: about 488.28 MB.
- Scalar int8 estimate: about 244-246 MB plus quantization scales/metadata.
- Report Docker container memory from `docker stats --no-stream`.
- Report average and p95 search latency from `synthetic_latency_summary.csv`.
- State the selected trade-off: float16 is the safest ranking-preserving optimization; scalar int8 is the more aggressive memory-saving option.
