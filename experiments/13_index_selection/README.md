# Index selection, hardware and production-like quantization

This stage closes the index-choice and deployment evidence gaps. It uses the same synthetic 500k vectors for every Qdrant variant and computes exact ground truth with FAISS `IndexFlatIP`.

## 1. Start the measured Qdrant server

Run from the repository root in PowerShell:

```powershell
docker compose up -d qdrant
docker inspect photographer-style-qdrant --format '{{.Config.Image}}'
```

Use server mode for the 500k run so Qdrant RAM is the container RAM:

```powershell
$env:QDRANT_MODE='server'
$env:QDRANT_URL='http://localhost:6333'
```

## 2. HNSW vs exact FAISS ground truth

The script recreates one collection per variant, uploads the same vectors, waits for green status, then records build time, Recall@10, p50/p95, disk size and Docker RAM. HNSW is explicitly fixed by `M`, `ef_construct`; query-time `ef_search` is also recorded.

```powershell
.\.venv\Scripts\python.exe experiments\scripts\benchmark_qdrant_index.py `
  --query-count 30 --latency-runs 5 --batch-size 2048
```

Outputs:

- `qdrant_index_benchmark.csv`
- `qdrant_index_benchmark.md`

Do not compare configurations from different corpus sizes or different query vectors. Use the Pareto choice with Recall@10 >= 0.98 unless the measured application requirement says otherwise.

## 3. Hardware snapshots at 25k and 500k

Capture snapshots only after the corresponding collection is fully uploaded and green. The command appends/replaces a labeled entry in one JSON file:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\capture_hardware_metrics.py --label 25k --collection photos
.\.venv\Scripts\python.exe experiments\scripts\capture_hardware_metrics.py --label 500k --collection index_benchmark_native_scalar_int8_m32_efc200_efs128
```

The JSON includes OS, CPU, host RAM, GPU/CUDA, Python, PyTorch, qdrant-client, Qdrant HTTP information, Qdrant storage disk size, and repeated `docker stats` samples. If host RAM is missing, install the optional observer dependency once:

```powershell
.\.venv\Scripts\python.exe -m pip install psutil
```

For a fair 25k measurement, create/upload the real `photos` collection and run the `--label 25k` command. For 500k, run it after the synthetic collection used by the benchmark is green. Keep the container name unchanged or pass `--container`.

## 4. Native Qdrant scalar INT8

`benchmark_qdrant_index.py` includes `native_scalar_int8_m32_efc200_efs128` by default. It uses Qdrant's server-side `ScalarQuantization(INT8)` and Qdrant search, so Python dequantization is not in latency.

To isolate only this production-like variant:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\benchmark_qdrant_index.py `
  --hnsw-config 32,200,128 --query-count 30 --latency-runs 5
```

Use its Recall@10, p50/p95, `container_memory_mb`, and disk columns beside the float32 HNSW baseline. The old `int8_per_vector_512` result in stage 12 remains a Python/Numpy diagnostic and must not be presented as native Qdrant latency.

## 5. Manual YOLO object validation

Regenerate the retrieval grids/template after Qdrant contains YOLO payloads:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\evaluate_yolo_object_retrieval.py
.\.venv\Scripts\python.exe experiments\scripts\prepare_yolo_visual_inspection.py
```

Open the PNG grids in `experiments\06_yolo_object_retrieval\visualizations`. In `visual_inspection.csv`, for the 12 object queries (`person`, `car`, `dog`, `cat`, `building`, `bird` and their six context variants) and modes `qdrant_semantic`, `qdrant_object`, `qdrant_object_rerank`, fill `object_present`:

- `1`: the requested object is visibly present;
- `0`: it is absent;
- leave no cells blank in these groups.

This is deliberately a visual label, not a copy of `detected_objects`. Run:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\compute_yolo_visual_metrics.py
```

The required comparison is in `object_precision_summary.csv` and `object_precision_metrics.csv`: semantic baseline vs object filter vs object-aware rerank. `visual_metrics.csv` remains the broader visual relevance evaluation and is only complete when its `visual_relevance` cells are also filled.

## Interpretation safeguards

- PCA rows are secondary diagnostics. They are not the main production recommendation until the low overlap@10 is explained with a separate quality analysis.
- Synthetic vectors are valid for index/scale/hardware comparison, not for claims about photograph relevance.
- Report both the exact ground truth and the Qdrant configuration; a latency number without Recall@10 is insufficient.
