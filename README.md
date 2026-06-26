# Photographer Style Search Engine

Photographer-oriented image retrieval built on CLIP embeddings, a FAISS exact-search baseline, Qdrant vector search, metadata filters, YOLO object detection, and style/object-aware reranking.

The project includes a script pipeline, FastAPI search API, Streamlit demo UI, local Qdrant path mode, optional Docker/server Qdrant mode, and reproducible experiment reports.

## What Is Implemented

- Unsplash Lite ingestion and local image download utilities
- SQLite metadata database at `data/metadata.sqlite`
- CLIP image and text encoding with OpenCLIP `ViT-B-32`
- Normalized `float32` embedding extraction into aligned NumPy arrays
- FAISS `IndexFlatIP` baseline for exact cosine-style retrieval
- Local persistent Qdrant collection named `photos`
- Image-to-image and text-to-image retrieval
- Visual descriptor extraction for brightness, contrast, saturation, warmth, and RGB color histograms
- YOLO object detection stored in SQLite and Qdrant payloads
- Style similarity and reranking over semantic candidates
- Object-aware reranking over semantic Qdrant candidates
- Qdrant payload filtering by keyword, detected object, and style ranges
- FastAPI endpoints for health, stats, text search, image search, metadata lookup, and local image serving
- Streamlit demo UI that calls the FastAPI API
- Optional Docker Qdrant support via `docker-compose.yml`
- Visualization scripts for FAISS and Qdrant search results
- Reproducible experiment reports under `experiments/`

Current local generated artifacts contain:

```text
clip_embeddings.shape = (24916, 512)
image_ids.shape       = (24916,)
SQLite image rows     = 24916
```

Generated data is intentionally ignored by Git. A fresh clone needs the dataset and pipeline steps below before search scripts can run.

## Architecture

### FAISS Baseline

```text
Query image or text
    -> CLIPEncoder
    -> normalized 512-d embedding
    -> FAISS IndexFlatIP
    -> top-k semantic candidates
    -> optional style reranking for image queries
    -> ranked results
```

### Qdrant Backend

```text
Query image or text
    -> CLIPEncoder
    -> normalized 512-d embedding
    -> Qdrant cosine vector search
       + optional keyword filter
       + optional detected-object filter
       + optional style range filters
    -> optional style reranking
    -> optional object-aware reranking for text queries
    -> ranked results with payload metadata
```

Qdrant payloads include image IDs, paths, selected Unsplash metadata, visual descriptors, keyword lists from `keywords.csv000`, and YOLO `detected_objects` copied from SQLite.

### Application Layer

```text
User / Streamlit demo
    -> FastAPI
    -> CLIP encoder
    -> Qdrant
    -> metadata/style/object filters
    -> rerankers
    -> image results
```

## Repository Layout

```text
Photographer-Style-Search-Engine/
|-- app/
|   |-- api/
|   |   |-- __init__.py
|   |   |-- dependencies.py
|   |   |-- routes.py
|   |   \-- schemas.py
|   |-- demo/
|   |   \-- streamlit_app.py
|   |-- database/
|   |   |-- __init__.py
|   |   \-- create_database.py
|   |-- ml/
|   |   |-- __init__.py
|   |   |-- clip_encoder.py
|   |   \-- visual_descriptor.py
|   |-- search/
|   |   |-- __init__.py
|   |   |-- faiss_index.py
|   |   |-- metadata_repository.py
|   |   |-- object_aware_reranker.py
|   |   |-- qdrant_config.py
|   |   |-- qdrant_retrieval_service.py
|   |   |-- qdrant_store.py
|   |   |-- retrieval_service.py
|   |   |-- style_reranker.py
|   |   |-- style_similarity.py
|   |   \-- vector_store.py
|   |-- __init__.py
|   \-- main.py
|-- data/
|   |-- embeddings/
|   |   |-- .gitkeep
|   |   |-- clip_embeddings.npy
|   |   \-- image_ids.npy
|   |-- indexes/
|   |   |-- .gitkeep
|   |   \-- flat.index
|   |-- qdrant/
|   |-- metadata.sqlite
|   \-- unsplash-lite/
|       |-- images/
|       |-- metadata.csv
|       |-- photos.csv000
|       |-- keywords.csv000
|       \-- other Unsplash Lite source files
|-- experiments/
|   |-- 01_clip_baseline/
|   |-- 02_style_reranking/
|   |-- 03_qdrant_backend/
|   |-- 04_filtered_retrieval/
|   |-- 05_scaled_retrieval_quality/
|   |-- 06_yolo_object_retrieval/
|   |-- configs/
|   |-- final_report/
|   |-- scripts/
|   |   |-- __init__.py
|   |   |-- compare_faiss_vs_qdrant.py
|   |   |-- compare_filtered_retrieval.py
|   |   |-- evaluate_reranking.py
|   |   |-- evaluate_scaled_retrieval_quality.py
|   |   |-- prepare_yolo_visual_inspection.py
|   |   |-- compute_yolo_visual_metrics.py
|   |   |-- evaluate_yolo_object_retrieval.py
|   |   \-- evaluate_style_reranking.py
|   |-- INDEX.md
|   |-- paths.py
|   \-- run_all_experiments.py
|-- scripts/
|   |-- build_flat_index.py
|   |-- compare_reranking.py
|   |-- extract_clip_embeddings.py
|   |-- extract_detected_objects.py
|   |-- extract_visual_features.py
|   |-- check_detected_objects.py
|   |-- migrate_add_detected_objects.py
|   |-- qdrant_common.py
|   |-- rebuild_metadata_csv.py
|   |-- setup_qdrant_collection.py
|   |-- test_clip_encoder.py
|   |-- test_image_search.py
|   |-- test_qdrant_image_search.py
|   |-- test_qdrant_object_search.py
|   |-- test_qdrant_text_search.py
|   |-- test_text_search.py
|   |-- upload_embeddings_to_qdrant.py
|   |-- visualize_qdrant_image_search.py
|   |-- visualize_qdrant_object_search.py
|   |-- visualize_qdrant_text_search.py
|   |-- visualize_search_results.py
|   \-- visualize_text_search.py
|-- dataset_check.py
|-- download_dataset.py
|-- requirements.txt
|-- docker-compose.yml
|-- LICENSE
\-- README.md
```

`docss/` exists in this workspace as ignored local planning/documentation material and is not required by the runnable pipeline.

## Setup

Target runtime is Python 3.12.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The dependency set includes OpenCLIP, PyTorch, FAISS CPU, Qdrant client, Ultralytics YOLO, OpenCV, Pillow, pandas, FastAPI/Uvicorn, Streamlit, requests, pytest, and supporting libraries.

## Dataset

Place the Unsplash Lite source files under:

```text
data/unsplash-lite/
```

At minimum, the pipeline expects:

```text
data/unsplash-lite/photos.csv000
data/unsplash-lite/keywords.csv000
```

Optional source files such as `collections.csv000`, `colors.csv000`, `conversions.csv000`, `DOCS.md`, `README.md`, and `TERMS.md` can remain in the same directory.

Inspect source columns:

```bash
python dataset_check.py
```

Download and prepare local images plus metadata:

```bash
python download_dataset.py
```

`download_dataset.py` targets up to 25,000 images, stores JPEGs in `data/unsplash-lite/images/`, and writes `data/unsplash-lite/metadata.csv`.

There is also an older helper, `scripts/rebuild_metadata_csv.py`, that rebuilds `data/metadata.csv` from `data/images/`. The main pipeline uses `data/unsplash-lite/metadata.csv`; use the rebuild helper only if you intentionally keep images in the older `data/images/` layout.

## Data Pipeline

Run these steps from the repository root.

Create or refresh the SQLite database:

```bash
python app/database/create_database.py
```

This refreshes SQLite from the metadata CSV. If you reset the database, rerun visual descriptors and YOLO detection before uploading Qdrant payloads again.

Extract visual descriptors into SQLite:

```bash
python scripts/extract_visual_features.py
```

Add YOLO object-detection columns to SQLite:

```bash
python scripts/migrate_add_detected_objects.py
```

Extract detected objects into SQLite:

```bash
python scripts/extract_detected_objects.py --model yolov8n.pt --device auto
```

Useful long-running options:

```bash
python scripts/extract_detected_objects.py --limit 100 --device cpu
python scripts/extract_detected_objects.py --offset 10000 --batch-size 32 --device auto
python scripts/extract_detected_objects.py --overwrite --confidence 0.25
python scripts/check_detected_objects.py
```

Extract normalized CLIP embeddings:

```bash
python scripts/extract_clip_embeddings.py
```

Build the FAISS index:

```bash
python scripts/build_flat_index.py
```

Create or recreate the Qdrant collection in local path mode:

```bash
$env:QDRANT_MODE="local"
python scripts/setup_qdrant_collection.py
```

Upload embeddings and metadata payloads into Qdrant:

```bash
python scripts/upload_embeddings_to_qdrant.py
```

Expected generated artifacts:

```text
data/metadata.sqlite
data/embeddings/clip_embeddings.npy
data/embeddings/image_ids.npy
data/indexes/flat.index
data/qdrant/
```

Local Qdrant path mode uses a filesystem lock, so run Qdrant scripts one at a time.

### Qdrant Modes

Default local mode stores Qdrant data under `data/qdrant/`:

```powershell
$env:QDRANT_MODE="local"
$env:QDRANT_PATH="data/qdrant"
.\.venv\Scripts\python.exe scripts\setup_qdrant_collection.py
.\.venv\Scripts\python.exe scripts\upload_embeddings_to_qdrant.py
```

Optional Docker/server mode stores Qdrant data under `data/qdrant_storage/`:

```powershell
docker compose up -d qdrant
$env:QDRANT_MODE="server"
$env:QDRANT_URL="http://localhost:6333"
.\.venv\Scripts\python.exe scripts\setup_qdrant_collection.py
.\.venv\Scripts\python.exe scripts\upload_embeddings_to_qdrant.py
.\.venv\Scripts\python.exe scripts\test_qdrant_text_search.py --query "dog on beach" --object dog --top-k 5
```

Supported Qdrant environment variables:

```text
QDRANT_MODE=local
QDRANT_PATH=data/qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=photos
```

## Search Usage

### FAISS

Image-to-image smoke test:

```bash
python scripts/test_image_search.py --image-id oSf8ePoG9NU --top-k 10
```

If `--image-id` is omitted, the script uses a deterministic first database row.

Text-to-image smoke test:

```bash
python scripts/test_text_search.py --query "warm cinematic landscape" --top-k 10
```

Visualize image search:

```bash
python scripts/visualize_search_results.py --image-id oSf8ePoG9NU --top-k 5
python scripts/visualize_search_results.py --image-id oSf8ePoG9NU --top-k 5 --rerank
python scripts/visualize_search_results.py --image-path path/to/query.jpg --top-k 5
```

Visualize text search:

```bash
python scripts/visualize_text_search.py --query "dark moody forest" --top-k 10
```

Compare CLIP-only image retrieval against style-reranked retrieval:

```bash
python scripts/compare_reranking.py --image-id oSf8ePoG9NU --top-k 10
python scripts/compare_reranking.py --image-path path/to/query.jpg --top-k 10
```

### Qdrant

Qdrant text search:

```bash
python scripts/test_qdrant_text_search.py --query "warm cinematic landscape" --top-k 10
python scripts/test_qdrant_text_search.py --query "street photography" --keyword person
python scripts/test_qdrant_text_search.py --query "person in street photography" --object person
python scripts/test_qdrant_text_search.py --query "dark forest" --max-brightness 0.4
python scripts/test_qdrant_text_search.py --query "beach sunset" --min-warmth 0.6 --min-saturation 0.45 --rerank
```

Object-aware Qdrant text search:

```bash
python scripts/test_qdrant_object_search.py --query "person in street photography" --object person
python scripts/test_qdrant_object_search.py --query "car at night" --object car --object-rerank
python scripts/test_qdrant_object_search.py --query "dog on beach" --object dog --candidate-pool-size 100 --object-rerank
```

Qdrant image search:

```bash
python scripts/test_qdrant_image_search.py --image-id oSf8ePoG9NU --top-k 10
python scripts/test_qdrant_image_search.py --image-path path/to/query.jpg --keyword nature
python scripts/test_qdrant_image_search.py --image-id oSf8ePoG9NU --rerank --candidate-pool-size 100
```

Qdrant visualization:

```bash
python scripts/visualize_qdrant_text_search.py --query "warm cinematic landscape" --top-k 5
python scripts/visualize_qdrant_text_search.py --query "street photography" --keyword person --top-k 5
python scripts/visualize_qdrant_text_search.py --query "person in street photography" --object person --top-k 5
python scripts/visualize_qdrant_object_search.py --query "car at night" --object car --object-rerank --top-k 5
python scripts/visualize_qdrant_image_search.py --image-id oSf8ePoG9NU --top-k 5
python scripts/visualize_qdrant_image_search.py --image-path path/to/query.jpg --keyword nature --top-k 5
```

Shared Qdrant filters:

```text
--keyword TEXT
--object TEXT
--rerank
--candidate-pool-size N
--min-brightness FLOAT
--max-brightness FLOAT
--min-contrast FLOAT
--max-contrast FLOAT
--min-saturation FLOAT
--max-saturation FLOAT
--min-warmth FLOAT
--max-warmth FLOAT
```

## API

Run the FastAPI app:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Health and stats:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/stats
```

Text search:

```powershell
$body = @{
  query = "dog on beach"
  top_k = 5
  candidate_pool_size = 100
  object = "dog"
  object_rerank = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/search/text `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Other endpoints:

```text
GET  /health
GET  /stats
POST /search/text
POST /search/image
GET  /images/{image_id}
GET  /image-file/{image_id}
```

## Demo UI

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Terminal 2:

```powershell
.\.venv\Scripts\streamlit.exe run app\demo\streamlit_app.py
```

## Experiments

The staged experiment layout is documented in `experiments/INDEX.md`.

Run individual stages:

```bash
.\.venv\Scripts\python.exe experiments\scripts\evaluate_style_reranking.py
.\.venv\Scripts\python.exe experiments\scripts\compare_faiss_vs_qdrant.py
.\.venv\Scripts\python.exe experiments\scripts\compare_filtered_retrieval.py
.\.venv\Scripts\python.exe experiments\scripts\evaluate_scaled_retrieval_quality.py
.\.venv\Scripts\python.exe experiments\scripts\evaluate_yolo_object_retrieval.py
.\.venv\Scripts\python.exe experiments\scripts\prepare_yolo_visual_inspection.py
.\.venv\Scripts\python.exe experiments\scripts\compute_yolo_visual_metrics.py
```

Regenerate all staged outputs:

```bash
.\.venv\Scripts\python.exe experiments\run_all_experiments.py
.\.venv\Scripts\python.exe experiments\run_all_experiments.py --assemble-only
```

Checked-in reports:

- `experiments/01_clip_baseline/report.md`
- `experiments/02_style_reranking/report.md`
- `experiments/03_qdrant_backend/report.md`
- `experiments/04_filtered_retrieval/report.md`
- `experiments/05_scaled_retrieval_quality/report.md`
- `experiments/06_yolo_object_retrieval/report.md`
- `experiments/final_report/results.md`

Current summarized findings:

- Style reranking improved average descriptor closeness in the 30-query evaluation for brightness, contrast, saturation, and warmth.
- FAISS and Qdrant matched exactly on unfiltered overlap@10 and top-1 consistency in the checked-in comparison.
- Qdrant adds useful retrieval behavior that FAISS does not provide here: keyword filters, style-range filters, persistent payloads, and filtered reranking.
- YOLO object payloads make object filters independent from noisy metadata keywords when detections are available.

## Core Module Reference

| Path | Purpose |
| --- | --- |
| `app/api/__init__.py` | API router export. |
| `app/api/dependencies.py` | Cached API search service, SQLite metadata lookup, stats, and image path resolution. |
| `app/api/routes.py` | FastAPI endpoints for health, stats, text/image search, metadata lookup, and image files. |
| `app/api/schemas.py` | Pydantic request and response models for the API. |
| `app/main.py` | FastAPI application entrypoint. |
| `app/demo/streamlit_app.py` | Streamlit demo UI that calls the FastAPI API. |
| `app/database/create_database.py` | SQLite schema creation and metadata import. |
| `app/ml/clip_encoder.py` | OpenCLIP model wrapper for normalized image and text embeddings. |
| `app/ml/visual_descriptor.py` | OpenCV/Pillow style descriptor extraction. |
| `app/search/faiss_index.py` | FAISS index build, save, load, and search wrapper. |
| `app/search/metadata_repository.py` | SQLite-backed metadata lookup and style feature parsing. |
| `app/search/object_aware_reranker.py` | Reranks semantic Qdrant text candidates with object, keyword, and style evidence. |
| `app/search/qdrant_config.py` | Environment-driven Qdrant local/server configuration. |
| `app/search/qdrant_retrieval_service.py` | Qdrant text/image retrieval service with filters and reranking. |
| `app/search/qdrant_store.py` | Qdrant collection, upload, keyword loading, and filter construction. |
| `app/search/retrieval_service.py` | FAISS-backed retrieval service. |
| `app/search/style_reranker.py` | Combines semantic and style scores for reranking. |
| `app/search/style_similarity.py` | Style similarity scoring for descriptors and histograms. |
| `app/search/vector_store.py` | Loader and validator for embedding and image ID arrays. |
| `app/search/__init__.py` | Public exports for search components. |

## Script Reference

| Path | Purpose |
| --- | --- |
| `dataset_check.py` | Prints a small sample and column list from `photos.csv000`. |
| `download_dataset.py` | Downloads Unsplash Lite images and writes main metadata CSV. |
| `app/database/create_database.py` | Builds SQLite tables and imports `data/unsplash-lite/metadata.csv`. |
| `scripts/extract_visual_features.py` | Computes visual style descriptors and stores them in SQLite. |
| `scripts/migrate_add_detected_objects.py` | Adds YOLO detection columns to the SQLite `images` table. |
| `scripts/extract_detected_objects.py` | Runs YOLO over local images and stores normalized detected-object labels in SQLite. |
| `scripts/check_detected_objects.py` | Prints SQLite object-detection coverage and top detected classes. |
| `scripts/extract_clip_embeddings.py` | Encodes local images with CLIP and writes embedding arrays. |
| `scripts/build_flat_index.py` | Builds `data/indexes/flat.index` from saved embeddings. |
| `scripts/setup_qdrant_collection.py` | Recreates the local Qdrant `photos` collection. |
| `scripts/upload_embeddings_to_qdrant.py` | Uploads vectors and metadata payloads to Qdrant. |
| `scripts/qdrant_common.py` | Shared Qdrant paths, service construction, query helpers, and filter CLI args. |
| `scripts/test_clip_encoder.py` | Smoke test for the OpenCLIP encoder. |
| `scripts/test_image_search.py` | FAISS image-to-image retrieval smoke test. |
| `scripts/test_text_search.py` | FAISS text-to-image retrieval smoke test. |
| `scripts/test_qdrant_image_search.py` | Qdrant image-to-image retrieval smoke test. |
| `scripts/test_qdrant_text_search.py` | Qdrant text-to-image retrieval smoke test. |
| `scripts/test_qdrant_object_search.py` | Qdrant text search with strict object filters or object-aware reranking. |
| `scripts/visualize_search_results.py` | Builds FAISS image search result grids. |
| `scripts/visualize_text_search.py` | Builds FAISS text search result grids. |
| `scripts/visualize_qdrant_image_search.py` | Builds Qdrant image search result grids. |
| `scripts/visualize_qdrant_text_search.py` | Builds Qdrant text search result grids. |
| `scripts/visualize_qdrant_object_search.py` | Builds object-filtered or object-aware Qdrant text search grids. |
| `scripts/compare_reranking.py` | Creates side-by-side CLIP-only versus reranked image grids. |
| `scripts/rebuild_metadata_csv.py` | Legacy metadata CSV rebuild helper for `data/images/`. |

## Experiment File Reference

| Path | Purpose |
| --- | --- |
| `experiments/INDEX.md` | Index of experiment stages, commands, outputs, and status. |
| `experiments/paths.py` | Centralized experiment directory paths and directory creation. |
| `experiments/run_all_experiments.py` | Runs staged visualizations/evaluations and assembles the final report. |
| `experiments/scripts/evaluate_reranking.py` | Main FAISS style-reranking evaluation implementation. |
| `experiments/scripts/evaluate_style_reranking.py` | Thin entrypoint that calls `evaluate_reranking.py`. |
| `experiments/scripts/compare_faiss_vs_qdrant.py` | Measures FAISS and Qdrant latency, overlap, consistency, and storage size. |
| `experiments/scripts/compare_filtered_retrieval.py` | Compares Qdrant semantic, keyword-filtered, style-filtered, and reranked modes. |
| `experiments/scripts/evaluate_scaled_retrieval_quality.py` | Evaluates scaled Qdrant retrieval quality, payload coverage, latency, weak relevance metrics, and qualitative failure modes. |
| `experiments/scripts/evaluate_yolo_object_retrieval.py` | Evaluates YOLO object payloads, object filters, keyword/object combinations, and object-aware reranking. |
| `experiments/scripts/prepare_yolo_visual_inspection.py` | Creates/preserves stage 06 manual visual inspection labels and writes the labeling guide. |
| `experiments/scripts/compute_yolo_visual_metrics.py` | Computes stage 06 visual metrics from complete manual top-10 labels and updates the report. |
| `experiments/01_clip_baseline/` | Checked-in CLIP-only baseline report and CSV. |
| `experiments/02_style_reranking/` | Checked-in style-reranking metrics, log, and report. |
| `experiments/03_qdrant_backend/` | Checked-in FAISS vs Qdrant CSV, markdown table, and report. |
| `experiments/04_filtered_retrieval/` | Checked-in filtered retrieval CSV, markdown table, and report. |
| `experiments/05_scaled_retrieval_quality/` | Scaled retrieval quality report, diagnostics, metrics CSVs, latency CSV, qualitative findings, and PNG grids. |
| `experiments/06_yolo_object_retrieval/` | YOLO object retrieval report, object payload stats, metrics CSVs, visual inspection template, qualitative findings, and PNG grids. |
| `experiments/final_report/` | Roll-up markdown report that links staged outputs. |
| `experiments/configs/.gitkeep` | Placeholder for future experiment configuration files. |

## Notes

- `image_ids.npy` is stored in the same row order as `clip_embeddings.npy`.
- CLIP embeddings are normalized and stored as `float32`.
- FAISS uses inner product search over normalized vectors, which behaves like cosine similarity.
- Qdrant uses cosine distance in either local path mode at `data/qdrant/` or server mode via `QDRANT_URL`.
- Text style reranking in Qdrant is heuristic. It only activates when `--rerank` is passed and the query contains style cues such as `dark`, `warm`, `cold`, `vibrant`, `muted`, `cinematic`, or `dramatic`.
- Visualization outputs are written under each stage's `visualizations/` directory or to an explicit `--output` path.
- Stage 06 visual metrics require manual labels in `experiments/06_yolo_object_retrieval/visual_inspection.csv`.
- YOLOv8n is CPU-expensive on the full corpus and is limited to COCO classes.
- Local Qdrant mode is convenient for development; Docker/server Qdrant is preferred for larger or shared runs.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
