# Photographer Style Search Engine: Final Report

## 1. Project Goal

This project is a photographer-oriented visual search engine for retrieving images by semantic content, visual style, metadata filters, and detected objects. It starts from CLIP semantic search and progressively adds style descriptors, Qdrant payload filtering, scaled diagnostics, YOLO object evidence, object-aware reranking, and a local application layer.

## 2. System Overview

- SQLite stores local image metadata, paths, descriptions, visual descriptors, and YOLO detection fields.
- OpenCLIP `ViT-B-32` produces normalized 512-dimensional image and text embeddings.
- FAISS `IndexFlatIP` remains the exact local baseline.
- Qdrant stores the real embeddings with payload fields for keywords, style descriptors, and detected objects.
- A separate Qdrant collection named `photos_synthetic_500k` stores generated synthetic vector objects for scalability benchmarking only.
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

- Real corpus: 24,916 downloaded images used for visual retrieval quality evaluation.
- Real CLIP embeddings: 24,916 x 512
- FAISS index vectors: 24,916
- Real Qdrant collection: `photos`
- Real Qdrant points: 24916
- YOLO object coverage: 0.4583
- Images with detected objects: 11418
- Unique detected classes: 78
- Synthetic corpus: 500000 generated vector objects used for scale, indexing, latency, and hardware requirement evaluation.
- Synthetic Qdrant collection: `photos_synthetic_500k`

## 4. Experiment Timeline

- `01_clip_baseline`: CLIP-only text retrieval examples.
- `02_style_reranking`: FAISS image retrieval plus style-aware reranking.
- `03_qdrant_backend`: Qdrant backend comparison against FAISS.
- `04_filtered_retrieval`: metadata and style filter evaluation.
- `05_scaled_retrieval_quality`: 24,916-image scale test and visual diagnosis of keyword noise.
- `06_yolo_object_retrieval`: YOLO object payloads, strict object filters, keyword/object combinations, and object-aware reranking.
- `07_synthetic_500k_scale`: synthetic 500k vector-object benchmark in a separate Qdrant collection for scalability and indexing tests.
- `10_relevance_labeling` and `11_search_quality_metrics`: shared human relevance judgments and ranking metrics.
- `12_vector_optimization`: float16, int8, and reduced-dimension vector optimization experiments for memory/latency trade-offs.
- `13_index_selection`: exact FAISS ground truth, explicit Qdrant HNSW configurations, native scalar INT8, and hardware snapshots.

## 5. Main Results

Scale:

Real corpus:

- 24,916 downloaded images
- used for visual retrieval quality
- 24,916 embeddings
- 24,916 Qdrant points in `photos`

Synthetic corpus:

- 500000 vector objects
- used for scale, indexing, latency, hardware requirement evaluation
- stored separately in `photos_synthetic_500k`

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

Stage 07:

- synthetic collection size = 500000
- qdrant_synthetic_semantic avg latency = 31.37 ms
- qdrant_synthetic_semantic p95 latency = 44.07 ms
- synthetic results are scalability/indexing diagnostics, not visual relevance conclusions.

Human relevance evaluation:

- labeled unique judgments = 478
- qdrant_filtered P@10 = 0.9464
- qdrant_filtered nDCG@10 = 0.9481
- qdrant_semantic P@10 = 0.9214
- qdrant_semantic nDCG@10 = 0.9134
- qdrant_object_rerank P@10 = 0.9464
- qdrant_object_rerank nDCG@10 = 0.9393
- human-label metrics are complete for the 28-query validation set.

Vector optimization:

- float16_512 estimated vector memory at 500k = 488.28 MB
- float16_512 memory reduction = 2.00x
- float16_512 overlap@10 vs baseline = 0.9964
- int8_per_vector_512 estimated vector memory at 500k = 246.05 MB
- int8_per_vector_512 memory reduction = 3.97x
- int8_per_vector_512 overlap@10 vs baseline = 0.8750

Index selection and hardware:

- hardware OS = Windows-10-10.0.19045-SP0; CPU logical cores = 16; host RAM = 16108.87 MB
- GPU/CUDA available = False; Python = 3.12.10; PyTorch = 2.12.0; Qdrant server = 1.18.2
- first explicit HNSW configuration Recall@10 = 1.0000, p50 = 16.02 ms, p95 = 31.98 ms
- native Qdrant scalar INT8 Recall@10 = 0.8700, p50 = 15.62 ms, p95 = 30.74 ms
- native scalar INT8 clean container RAM = 1475.58 MB; disk = 1734.37 MB
- Docker RAM snapshot 25k = 95.27 MB; disk = 624.76 MB

Manual YOLO object validation:

- complete object-label query groups = 12
- Object Precision@10 semantic baseline = 0.9167
- Semantic 95% bootstrap CI = [0.8250, 0.9917]
- Object Precision@10 object filter = 0.9800
- Object-filter delta vs semantic = 0.0633
- Object-filter label coverage = 10/12 groups, 100/120 images
- Object-filter 95% bootstrap CI = [0.9500, 1.0000]
- Object-filter empty result queries = 2 (0.1667)
- Object Precision@10 object-aware rerank = 0.9833
- Rerank delta vs semantic = 0.0667
- Rerank label coverage = 12/12 groups, 120/120 images
- Rerank 95% bootstrap CI = [0.9583, 1.0000]

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
- Manual visual inspection is reported separately from automatic YOLO payload metrics; `object_precision_summary.csv` is only populated for complete `object_present` labels.
- YOLOv8n uses the fixed COCO class set and misses open-vocabulary scene concepts.
- Local Qdrant path mode is convenient but not ideal above 20k points.
- CPU YOLO inference is slow for full-corpus extraction.
- Synthetic 500k vectors are generated from real embeddings and are not independent real photographs.
- Synthetic payloads copy metadata from source images, so synthetic results cannot be used for visual relevance claims.
- App-level float16/int8 benchmark latencies include NumPy conversion overhead; native Qdrant scalar INT8 results in stage 13 are the production-like comparison.
- The frontend is a local demo, not a production product UI.

## 9. Future Work

- Complete any remaining manual visual inspection groups in stage 06.
- Tune the YOLO confidence threshold.
- Store bounding boxes and confidence scores.
- Use Docker/server Qdrant for larger runs.
- Add open-vocabulary detection with GroundingDINO or OWL-ViT.
- Improve and deploy the frontend.
- Deploy the API with a persistent Qdrant server.

## 10. Conclusion

The project has evolved from a CLIP-only retrieval baseline into a multi-signal image search system. It combines semantic embeddings, exact FAISS comparison, Qdrant payload filtering, metadata keywords, visual style descriptors, YOLO object detections, and reranking strategies. The project contains a 24,916-image real visual corpus for quality evaluation and a 500,000-object synthetic vector corpus for scalability and indexing evaluation.

## Reproduce Report Assembly

```powershell
.\.venv\Scripts\python.exe experiments\run_all_experiments.py --assemble-only
```