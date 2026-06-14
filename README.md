# Photographer Style Search Engine

Photographer-oriented image retrieval built on top of CLIP embeddings, FAISS semantic search, and a second-stage photographic style reranker.

## Current Stage

The repository now contains a complete two-stage retrieval pipeline:

1. Semantic retrieval with CLIP + FAISS
2. Style-aware reranking with brightness, contrast, saturation, warmth, and color histogram descriptors

Implemented components:

- Unsplash Lite subset and local image files
- SQLite metadata database in `data/metadata.sqlite`
- Visual descriptor extraction into SQLite
- CLIP image and text encoder with OpenCLIP
- Bulk CLIP embedding extraction into aligned NumPy arrays
- Persistent FAISS Flat index
- Image-to-image retrieval
- Text-to-image retrieval
- Style similarity engine
- Style reranker
- Reranking evaluation scripts
- Search result visualization and CLIP-vs-reranked comparison images

## Retrieval Architecture

### Stage 1: Semantic retrieval

```text
Query image / text
    -> CLIP encoder
    -> 512-d embedding
    -> FAISS Flat index
    -> Top-K semantic candidates
```

### Stage 2: Style-aware reranking

```text
Top semantic candidates
    -> style similarity engine
    -> combined semantic + style score
    -> final ranked results
```

For image queries, reranking uses descriptors already stored in SQLite:

- `brightness`
- `contrast`
- `saturation`
- `warmth`
- `color_histogram`

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
|       |-- retrieval_service.py
|       |-- style_reranker.py
|       |-- style_similarity.py
|       \-- vector_store.py
|-- scripts/
|   |-- build_flat_index.py
|   |-- compare_reranking.py
|   |-- extract_clip_embeddings.py
|   |-- extract_visual_features.py
|   |-- test_clip_encoder.py
|   |-- test_image_search.py
|   |-- test_text_search.py
|   \-- visualize_search_results.py
|-- experiments/
|   |-- evaluate_reranking.py
|   \-- evaluate_style_reranking.py
|-- data/
|   |-- embeddings/
|   |   |-- clip_embeddings.npy
|   |   \-- image_ids.npy
|   |-- indexes/
|   |   \-- flat.index
|   |-- metadata.sqlite
|   \-- unsplash-lite/
|-- PROMPTS/
|-- README.md
\-- requirements.txt
```

## Setup

Target runtime is Python 3.12.

Create or activate the local virtual environment:

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

Expected core artifacts:

```text
data/metadata.sqlite
data/embeddings/clip_embeddings.npy
data/embeddings/image_ids.npy
data/indexes/flat.index
```

For the current 1000-image subset:

```python
clip_embeddings.shape == (1000, 512)
image_ids.shape == (1000,)
```

## Search Usage

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

## Evaluation

Run the reranking evaluation:

```bash
python experiments/evaluate_style_reranking.py
```

This reports average query-to-result differences for:

- brightness
- contrast
- saturation
- warmth

The current reranking stage produces measurable improvement over CLIP-only retrieval on the local sample set.

## Notes

- `image_ids.npy` is stored in the same order as `clip_embeddings.npy`
- CLIP embeddings are normalized and stored as `float32`
- FAISS uses `IndexFlatIP`, so normalized embeddings behave as cosine similarity
- Text queries use semantic retrieval only; image queries can be reranked by photographic style
- Generated visualization images are written to `experiments/visualizations/`
