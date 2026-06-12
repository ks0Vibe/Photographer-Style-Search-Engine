# Photographer Style Search Engine

A photographer-oriented visual search project that combines semantic CLIP embeddings with handcrafted photographic style descriptors.

## Current Stage

The repository currently includes:

- Unsplash Lite metadata and downloaded images
- A SQLite metadata database
- Visual descriptor extraction for brightness, contrast, saturation, warmth, and color histogram
- A reusable CLIP encoder for images and text
- Bulk CLIP embedding extraction into aligned NumPy files

## Pipeline

Current processing flow:

1. Prepare image metadata and downloaded files
2. Build `data/metadata.sqlite`
3. Extract visual style descriptors into SQLite
4. Extract normalized CLIP embeddings into `data/embeddings/`

Planned next step:

1. Build a FAISS index from the saved CLIP embeddings
2. Add semantic retrieval plus style-aware reranking

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
|-- scripts/
|   |-- extract_clip_embeddings.py
|   |-- extract_visual_features.py
|   |-- rebuild_metadata_csv.py
|   \-- test_clip_encoder.py
|-- data/
|   |-- embeddings/
|   |   |-- clip_embeddings.npy
|   |   \-- image_ids.npy
|   |-- indexes/
|   |-- metadata.sqlite
|   \-- unsplash-lite/
|-- experiments/
|-- PROMPT.md
|-- README.md
\-- requirements.txt
```

## Setup

Use Python 3.12 and the project virtual environment when working with `open_clip`.

```bash
pip install -r requirements.txt
```

If you are using the local virtual environment in this repository:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Usage

Build the metadata database:

```bash
python app/database/create_database.py
```

Extract visual descriptors:

```bash
python scripts/extract_visual_features.py
```

Extract CLIP embeddings:

```bash
python scripts/extract_clip_embeddings.py
```

Expected outputs after CLIP extraction:

```text
data/embeddings/clip_embeddings.npy
data/embeddings/image_ids.npy
```

For a 1000-image dataset, the expected shapes are:

```python
(1000, 512)
(1000,)
```

## Notes

- `image_ids.npy` is stored in the same order as `clip_embeddings.npy`
- CLIP embeddings are normalized and saved as `float32`
- Semantic embeddings and style descriptors are intended to be combined in the later retrieval pipeline
