# Photographer Style Search Engine

DeepStyle is a visual search engine for photographers, designers, and content teams. It finds image references not only by objects and keywords, but also by semantic meaning, mood, color, lighting, and visual style.

The project combines CLIP embeddings, FAISS and Qdrant vector search, Unsplash metadata, YOLO object detection, and style/object-aware reranking.

## Features

- text-to-image semantic search;
- image-to-image search;
- keyword, object, and style filters;
- CLIP-based vector similarity;
- YOLO detected-object metadata;
- style-aware and object-aware reranking;
- FastAPI backend;
- Streamlit demo interface.

## Architecture

```text
User
  -> Streamlit UI
  -> FastAPI
  -> CLIP encoder
  -> Qdrant vector search
       + keyword/object/style filters
       + optional reranking
  -> ranked image results
```

### Offline indexing

```text
Unsplash images and metadata
  -> SQLite metadata database
  -> visual descriptors
  -> CLIP image embeddings
  -> YOLO object detection
  -> FAISS index and Qdrant collection
```

### Main components

| Component | Purpose |
| --- | --- |
| OpenCLIP ViT-B-32 | Encodes text and images into normalized 512-dimensional vectors |
| FAISS IndexFlatIP | Exact local vector-search baseline |
| Qdrant | Vector database for production-like search and payload filters |
| SQLite | Image metadata, visual descriptors, and detected objects |
| YOLOv8n | Detects objects from the fixed COCO class set |
| FastAPI | Search API |
| Streamlit | Interactive demo UI |

## Project structure

```text
app/           Application, API, models, search services, and Streamlit UI
scripts/       Dataset preparation, indexing, upload, and smoke-test scripts
experiments/   Evaluation reports, validation data, and benchmarks
tests/         Automated tests
data/          Local dataset, embeddings, indexes, and Qdrant storage
```

## Requirements

- Python 3.12;
- 8 GB RAM minimum, 16 GB recommended for the full local corpus;
- Docker Desktop for Qdrant server mode;
- GPU is optional, but recommended for CLIP and YOLO preprocessing;
- the Unsplash Lite source files.

## Quick start

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Prepare the dataset

Place the Unsplash Lite source files in:

```text
data/unsplash-lite/
```

At minimum, the pipeline expects:

```text
data/unsplash-lite/photos.csv000
data/unsplash-lite/keywords.csv000
```

Download the local image corpus and create the metadata CSV:

```powershell
python scripts/download_dataset.py
```

If the dataset is already prepared, skip this step.

### 3. Build the search data

Run the commands from the repository root:

```powershell
python app/database/create_database.py
python scripts/extract_visual_features.py
python scripts/migrate_add_detected_objects.py
python scripts/extract_detected_objects.py --model yolov8n.pt --device auto
python scripts/extract_clip_embeddings.py
python scripts/build_flat_index.py
```

The main generated artifacts are:

```text
data/metadata.sqlite
data/embeddings/clip_embeddings.npy
data/embeddings/image_ids.npy
data/indexes/flat.index
```

### 4. Start Qdrant

For a simple local run, use filesystem-backed Qdrant:

```powershell
$env:QDRANT_MODE="local"
$env:QDRANT_PATH="data/qdrant"
python scripts/setup_qdrant_collection.py
python scripts/upload_embeddings_to_qdrant.py
```

For Docker/server mode:

```powershell
docker compose up -d qdrant
$env:QDRANT_MODE="server"
$env:QDRANT_URL="http://localhost:6333"
python scripts/setup_qdrant_collection.py
python scripts/upload_embeddings_to_qdrant.py
```

Run Qdrant setup and upload one at a time when using local filesystem mode.

### 5. Start the API

The project API uses port `8001` in the demo setup:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8001
```

Check that the API is available:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

### 6. Start the Streamlit demo

Open a second terminal:

```powershell
.\.venv\Scripts\streamlit.exe run app\demo\streamlit_app.py
```

The UI provides text search, image upload search, dataset image search by ID, filters, and reranking controls.

The API URL can be changed in the Streamlit sidebar or configured with:

```powershell
$env:DEEPSTYLE_API_URL="http://localhost:8001"
```

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | API health check |
| GET | `/stats` | Dataset and Qdrant statistics |
| POST | `/search/text` | Text-to-image search |
| POST | `/search/image` | Search by dataset image ID or local image path |
| POST | `/search/image/upload` | Search by uploaded image |
| GET | `/image-file/{image_id}` | Serve a result image |

Swagger documentation is available at:

```text
http://127.0.0.1:8001/docs
```

## Evaluation notes

The real visual corpus contains **24,916 images**. A separate synthetic corpus of **500,000 vectors** is used only for vector-database scalability and indexing benchmarks; it does not represent 500,000 real photographs.

Detailed experiment reports, validation results, human relevance metrics, and benchmark outputs are stored in [`experiments/`](experiments/).

The main reported systems are:

- `faiss_baseline` — exact FAISS semantic search;
- `faiss_style_rerank` — FAISS results with style reranking;
- `qdrant_semantic` — Qdrant semantic search;
- `qdrant_filtered` — Qdrant search with payload filters;
- `qdrant_object_rerank` — Qdrant candidates with object-aware reranking.

## Limitations

- Unsplash keywords may be noisy or incomplete;
- YOLOv8n is limited to the fixed COCO classes and does not provide open-vocabulary detection;
- CPU YOLO inference is slow for full-corpus processing;
- synthetic 500k vectors are suitable for scalability tests, not visual relevance conclusions;
- CLIP and YOLO use pretrained weights; the project does not fine-tune these models.

## Tests

Run the automated tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
