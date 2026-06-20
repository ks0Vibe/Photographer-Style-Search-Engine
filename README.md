# Photographer Style Search Engine

Photographer-oriented image retrieval built on top of CLIP embeddings, a FAISS baseline, and a Qdrant retrieval backend with payload filtering and style-aware reranking.

## Current Stage

The repository now contains two retrieval backends:

1. FAISS Flat baseline for exact semantic search
2. Qdrant local vector database for semantic search plus metadata-aware filtering

Implemented components:

- Unsplash Lite subset and local image files
- SQLite metadata database in `data/metadata.sqlite`
- Visual descriptor extraction into SQLite
- CLIP image and text encoder with OpenCLIP
- Bulk CLIP embedding extraction into aligned NumPy arrays
- Persistent FAISS Flat index
- Persistent local Qdrant collection
- Image-to-image retrieval
- Text-to-image retrieval
- Style similarity engine and style reranker
- Keyword-aware Qdrant retrieval
- Visualization scripts for FAISS and Qdrant
- Comparison reports for FAISS vs Qdrant and filtered retrieval

## Retrieval Architecture

### FAISS baseline

```text
Query image / text
    -> CLIP encoder
    -> 512-d embedding
    -> FAISS Flat index
    -> Top-K semantic candidates
    -> optional style reranking
```

### Qdrant backend

```text
Query image / text
    -> CLIP encoder
    -> 512-d embedding
    -> Qdrant vector search
       + optional keyword filter
       + optional object filter
       + optional style payload filters
    -> Top-K semantic candidates
    -> optional style reranking
    -> final ranked results
```

For image queries, reranking uses stored visual descriptors:

- `brightness`
- `contrast`
- `saturation`
- `warmth`
- `color_histogram`

Qdrant payloads store:

- CLIP embeddings
- file paths and selected metadata
- visual descriptors
- Unsplash keywords
- prepared `detected_objects` field

## Project Structure

```text
Photographer-Style-Search-Engine/
|-- app/
|   |-- database/
|   |   \-- create_database.py
|   |-- ml/
|   |   |-- clip_encoder.py
|   |   \-- visual_descriptor.py
|   \-- search/
|       |-- faiss_index.py
|       |-- metadata_repository.py
|       |-- qdrant_retrieval_service.py
|       |-- qdrant_store.py
|       |-- retrieval_service.py
|       |-- style_reranker.py
|       |-- style_similarity.py
|       \-- vector_store.py
|-- scripts/
|   |-- build_flat_index.py
|   |-- compare_reranking.py
|   |-- extract_clip_embeddings.py
|   |-- extract_visual_features.py
|   |-- qdrant_common.py
|   |-- setup_qdrant_collection.py
|   |-- test_image_search.py
|   |-- test_qdrant_image_search.py
|   |-- test_qdrant_text_search.py
|   |-- test_text_search.py
|   |-- upload_embeddings_to_qdrant.py
|   |-- visualize_qdrant_image_search.py
|   |-- visualize_qdrant_text_search.py
|   |-- visualize_search_results.py
|   \-- visualize_text_search.py
|-- experiments/
|   |-- 01_clip_baseline/
|   |-- 02_style_reranking/
|   |-- 03_qdrant_backend/
|   |-- 04_filtered_retrieval/
|   |-- final_report/
|   |-- scripts/
|   |-- INDEX.md
|   \-- run_all_experiments.py
|-- data/
|   |-- embeddings/
|   |   |-- clip_embeddings.npy
|   |   \-- image_ids.npy
|   |-- indexes/
|   |   \-- flat.index
|   |-- qdrant/
|   |-- metadata.sqlite
|   \-- unsplash-lite/
|-- docss/
|-- README.md
\-- requirements.txt
```

## Setup

Target runtime is Python 3.12.

Activate the local virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Pipeline

Build the metadata database:

```bash
python app/database/create_database.py
```

Extract visual descriptors into SQLite:

```bash
python scripts/extract_visual_features.py
```

Extract normalized CLIP embeddings:

```bash
python scripts/extract_clip_embeddings.py
```

Build the FAISS Flat index:

```bash
python scripts/build_flat_index.py
```

### Qdrant Setup

Create or recreate the local Qdrant collection:

```bash
python scripts/setup_qdrant_collection.py
```

Upload embeddings and payloads into Qdrant:

```bash
python scripts/upload_embeddings_to_qdrant.py
```

Expected core artifacts:

```text
data/metadata.sqlite
data/embeddings/clip_embeddings.npy
data/embeddings/image_ids.npy
data/indexes/flat.index
data/qdrant/
```

For the current 1000-image subset:

```python
clip_embeddings.shape == (1000, 512)
image_ids.shape == (1000,)
```

## Search Usage

### FAISS baseline

Image-to-image retrieval:

```bash
python scripts/test_image_search.py
```

Text-to-image retrieval:

```bash
python scripts/test_text_search.py --query "warm cinematic landscape"
```

Visualize image search results:

```bash
python scripts/visualize_search_results.py --image-id oSf8ePoG9NU --top-k 5
```

Visualize reranked results with score breakdowns:

```bash
python scripts/visualize_search_results.py --image-id oSf8ePoG9NU --top-k 5 --rerank
```

Generate a side-by-side CLIP-only vs reranked comparison:

```bash
python scripts/compare_reranking.py --image-id oSf8ePoG9NU --top-k 5
```

### Qdrant retrieval backend

Run Qdrant text search:

```bash
python scripts/test_qdrant_text_search.py --query "warm cinematic landscape"
python scripts/test_qdrant_text_search.py --query "street photography" --keyword person
python scripts/test_qdrant_text_search.py --query "dark forest" --max-brightness 0.4
```

Run Qdrant image search:

```bash
python scripts/test_qdrant_image_search.py --image-id oSf8ePoG9NU
python scripts/test_qdrant_image_search.py --image-id oSf8ePoG9NU --keyword nature
```

Visualize Qdrant text search:

```bash
python scripts/visualize_qdrant_text_search.py --query "warm cinematic landscape" --top-k 5
python scripts/visualize_qdrant_text_search.py --query "street photography" --keyword person --top-k 5
```

Visualize Qdrant image search:

```bash
python scripts/visualize_qdrant_image_search.py --image-id oSf8ePoG9NU --top-k 5
python scripts/visualize_qdrant_image_search.py --image-id oSf8ePoG9NU --keyword nature --top-k 5
```

## Evaluation

Run the FAISS style-reranking evaluation:

```bash
.\.venv\Scripts\python.exe experiments/scripts/evaluate_style_reranking.py
```

Run the FAISS vs Qdrant comparison:

```bash
.\.venv\Scripts\python.exe experiments/scripts/compare_faiss_vs_qdrant.py
```

Run the Qdrant filtered retrieval comparison:

```bash
.\.venv\Scripts\python.exe experiments/scripts/compare_filtered_retrieval.py
```

Regenerate the staged experiment outputs in one pass:

```bash
.\.venv\Scripts\python.exe experiments/run_all_experiments.py
```

Generated reports:

- `experiments/01_clip_baseline/report.md`
- `experiments/02_style_reranking/report.md`
- `experiments/03_qdrant_backend/report.md`
- `experiments/04_filtered_retrieval/report.md`
- `experiments/final_report/results.md`

## Qdrant Retrieval Backend

The project supports Qdrant as a vector database backend.

Qdrant stores:

- CLIP embeddings
- image metadata
- visual descriptors
- keywords
- detected objects

This enables filtered vector search and keeps FAISS available as a baseline for exact local comparison.

## Notes

- `image_ids.npy` is stored in the same order as `clip_embeddings.npy`
- CLIP embeddings are normalized and stored as `float32`
- FAISS uses `IndexFlatIP`, so normalized embeddings behave as cosine similarity
- Qdrant uses cosine distance in a persistent local collection at `data/qdrant/`
- Local Qdrant path mode uses a filesystem lock, so Qdrant scripts should be run one at a time
- Text queries can use semantic search alone or Qdrant payload filters
- `detected_objects` is prepared in payloads but not populated by an object-detection pipeline yet
- Experiment outputs are grouped by stage under `experiments/`; generated visualization images are written to each stage's `visualizations/` directory
