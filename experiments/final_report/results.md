# Photographer Style Search Engine: Final Report

## 1. Project Goal

This project is a photographer-oriented visual search engine for retrieving images by semantic content, visual style, metadata filters, and detected objects. It starts from CLIP semantic search and progressively adds style descriptors, Qdrant payload filtering, scaled diagnostics, YOLO object evidence, object-aware reranking, and a local application layer.

## 2. System Overview

- SQLite stores local image metadata, paths, descriptions, visual descriptors, and YOLO detection fields.
- OpenCLIP `ViT-B-32` produces normalized 512-dimensional image and text embeddings.
- FAISS `IndexFlatIP` remains the exact local baseline.
- Qdrant stores the same embeddings with payload fields for keywords, style descriptors, and detected objects.
- Unsplash keyword metadata supports broad filtering but can be noisy.
- Visual descriptors capture brightness, contrast, saturation, warmth, and color histograms.
- Style reranking improves visual consistency for style-heavy searches.
- YOLO detected objects add image-derived object evidence.
- Object-aware reranking combines semantic, object, keyword, and style signals without requiring a strict hard filter.

Application flow:

```text
User / Streamlit demo
    -> FastAPI
    -> CLIP encoder
    -> Qdrant
    -> metadata/style/object filters
    -> rerankers
    -> image results
```

## 3. Data and Artifacts

- Images: 24,916
- CLIP embeddings: 24,916 x 512
- FAISS index vectors: 24,916
- Qdrant collection: `photos`
- Qdrant points: 24916
- YOLO object coverage: 0.4583
- Images with detected objects: 11418
- Unique detected classes: 78

## 4. Experiment Timeline

- `01_clip_baseline`: CLIP-only text retrieval examples.
- `02_style_reranking`: FAISS image retrieval plus style-aware reranking.
- `03_qdrant_backend`: Qdrant backend comparison against FAISS.
- `04_filtered_retrieval`: metadata and style filter evaluation.
- `05_scaled_retrieval_quality`: 24,916-image scale test and visual diagnosis of keyword noise.
- `06_yolo_object_retrieval`: YOLO object payloads, strict object filters, keyword/object combinations, and object-aware reranking.

## 5. Main Results

Scale:

- 24,916 images
- 24,916 embeddings
- 24,916 Qdrant points

Stage 05:

- keyword coverage = 1.0000
- object coverage before YOLO = 0.0000
- visual inspection showed that metadata keyword filters help but can admit noisy object/scene matches.

Stage 06:

- object coverage after YOLO = 0.4583
- images with detected objects = 11418
- unique detected classes = 78
- qdrant_object avg relevance = 1.9100
- qdrant_keyword avg relevance = 1.6417
- qdrant_object_rerank nDCG@10 = 0.9795

## 6. Qualitative Findings

- Keyword filters help narrow the corpus, but they are noisy because they depend on external metadata.
- Style reranking improves style consistency, but it does not guarantee object presence.
- YOLO helps object-specific retrieval for COCO-supported classes such as person, dog, cat, bird, and car.
- Object-aware reranking is less brittle than strict filtering because it keeps semantic candidates and promotes object matches instead of excluding all misses.
- Building and architecture remain scene/open-vocabulary limitations because `building` is not a YOLOv8n COCO class.

Representative visual grids:

![dog_on_beach__qdrant_keyword_primary](selected_visualizations/dog_on_beach__qdrant_keyword_primary.png)

![person_in_street_photography__qdrant_keyword_primary](selected_visualizations/person_in_street_photography__qdrant_keyword_primary.png)

![dark_moody_forest__qdrant_rerank](selected_visualizations/dark_moody_forest__qdrant_rerank.png)

![minimal_architecture__qdrant_semantic](selected_visualizations/minimal_architecture__qdrant_semantic.png)

![person__qdrant_object](selected_visualizations/person__qdrant_object.png)

![car_at_night__qdrant_object](selected_visualizations/car_at_night__qdrant_object.png)

![dog_on_beach__qdrant_object_rerank](selected_visualizations/dog_on_beach__qdrant_object_rerank.png)

![bird_in_nature__qdrant_keyword_object](selected_visualizations/bird_in_nature__qdrant_keyword_object.png)

## 7. Application Layer

The project now includes a local FastAPI API and a Streamlit demo UI. The API exposes health, corpus stats, text search, image search, metadata lookup, and local image-file endpoints. Heavy objects such as the CLIP encoder and Qdrant service are cached through FastAPI dependencies so the model is not reloaded for every request. The Streamlit demo calls the API rather than duplicating retrieval logic.

## 8. Limitations

- Automatic relevance labels are weak diagnostics, not human ground truth.
- Manual visual inspection support exists, but stage 06 labels are still incomplete unless `visual_metrics.csv` contains rows.
- YOLOv8n uses the fixed COCO class set and misses open-vocabulary scene concepts.
- Local Qdrant path mode is convenient but not ideal above 20k points.
- CPU YOLO inference is slow for full-corpus extraction.
- The frontend is a local demo, not a production product UI.

## 9. Future Work

- Finish manual visual inspection for stage 06.
- Tune the YOLO confidence threshold.
- Store bounding boxes and confidence scores.
- Use Docker/server Qdrant for larger runs.
- Add open-vocabulary detection with GroundingDINO or OWL-ViT.
- Improve and deploy the frontend.
- Deploy the API with a persistent Qdrant server.

## 10. Conclusion

The project has evolved from a CLIP-only retrieval baseline into a multi-signal image search system. It combines semantic embeddings, exact FAISS comparison, Qdrant payload filtering, metadata keywords, visual style descriptors, YOLO object detections, and reranking strategies. The resulting system can search by content, style, metadata, and detected object evidence while keeping the experiment trail reproducible.

## Reproduce Report Assembly

```powershell
.\.venv\Scripts\python.exe experiments\run_all_experiments.py --assemble-only
```