# Query Groups

This validation set groups queries by the retrieval behavior they are meant to test. The same query can be run against multiple systems so final metrics can compare retrieval quality across stages.

## semantic_text

Semantic text queries check whether CLIP retrieves visually and semantically related images from natural language prompts. They focus on broad visual concepts such as landscapes, portraits, food, city scenes, and architecture.

Primary systems:

- FAISS CLIP baseline
- Qdrant semantic search

## style_text

Style text queries check whether the search system responds to visual style cues such as warmth, darkness, cinematic mood, saturation, contrast, color grading, and soft light. These queries are useful for comparing CLIP-only search against style-aware reranking.

Primary systems:

- FAISS CLIP baseline
- FAISS + style reranking
- Qdrant semantic search

## object_text

Object text queries check object retrieval and YOLO/object filter behavior. They include prompts such as people, dogs, cars, bicycles, birds, boats, cups, and trains in realistic visual contexts.

Primary systems:

- Qdrant semantic search
- Qdrant filtered search
- Qdrant + object-aware reranking

## mixed_text

Mixed text queries combine object, style, and scene constraints. They test whether the system can retrieve images that satisfy multiple conditions at once, such as a warm outdoor portrait or a cinematic car scene at night.

Primary systems:

- FAISS CLIP baseline
- FAISS + style reranking
- Qdrant semantic search
- Qdrant filtered search
- Qdrant + object-aware reranking

## image_to_image

Image-to-image queries check visual similarity to an existing dataset image. The placeholder query values in `queries.csv` should be replaced with real image IDs after sampling from the local indexed dataset.

Primary systems:

- FAISS CLIP baseline image search
- Qdrant semantic image search
- Qdrant + object-aware reranking when object payloads are available

## System Coverage

| Query group | FAISS CLIP baseline | FAISS + style reranking | Qdrant semantic search | Qdrant filtered search | Qdrant + object-aware reranking |
| --- | --- | --- | --- | --- | --- |
| semantic_text | Yes | Optional comparison | Yes | Optional comparison | Optional comparison |
| style_text | Yes | Yes | Yes | Optional comparison | Optional comparison |
| object_text | Optional comparison | No | Yes | Yes | Yes |
| mixed_text | Yes | Yes | Yes | Yes | Yes |
| image_to_image | Yes | No | Yes | Optional comparison | Optional comparison |
