# Scaled Retrieval Quality Evaluation

## 1. Goal

This evaluation tests retrieval behavior after scaling the local corpus to approximately 25k images. The goal is not only to verify that CLIP/Qdrant retrieval remains operational, but also to determine whether the returned images actually look relevant when inspected visually.

## 2. Current System

The system uses Unsplash Lite images, SQLite metadata, OpenCLIP `ViT-B-32` image/text embeddings, a FAISS `IndexFlatIP` baseline, and a local persistent Qdrant collection named `photos`. Qdrant payloads include metadata keywords and visual descriptors. Style-aware reranking uses brightness, contrast, saturation, warmth, and color histograms.

## 3. Dataset and Index Statistics

| stat | value |
| --- | --- |
| sqlite_image_rows | 24916 |
| embeddings_shape | [24916, 512] |
| image_ids_shape | [24916] |
| faiss_index_vectors | 24916 |
| qdrant_points | 24916 |
| embedding_dim | 512 |
| failed_embeddings_if_available | 0 |
| qdrant_collection | photos |
| qdrant_distance | Cosine |

The scaled pipeline is technically successful and operational at 24,916 images. All local metadata rows have embeddings and are indexed in FAISS and Qdrant.

## 4. Payload Diagnostics

| stat | value |
| --- | --- |
| keyword_coverage | 1.0000 |
| object_coverage | 0.0000 |
| unique_keywords | 23830 |
| unique_detected_objects | 0 |

Top payload keywords:

| keyword | count |
| --- | --- |
| nature | 17678 |
| outdoors | 16496 |
| plant | 12143 |
| landscape | 8416 |
| scenery | 8279 |
| tree | 8076 |
| animal | 7940 |
| water | 7823 |
| sky | 7082 |
| mountain | 6619 |
| land | 5756 |
| sea | 5540 |
| ocean | 5371 |
| building | 4895 |
| vegetation | 4597 |
| light | 4262 |
| person | 4110 |
| flower | 4091 |
| snow | 4067 |
| blossom | 3954 |
| human | 3905 |
| mountain range | 3887 |
| ice | 3886 |
| weather | 3823 |
| coast | 3675 |

Qdrant keyword coverage is 100%, but detected-object coverage is 0%. This means keyword search can narrow candidates by metadata, but it cannot verify whether an object is visually present or central. Metadata keywords are broad and over-inclusive; examples include `dog` on rabbit images, `food` on rabbit/plant/horse images, `portrait` on people/dogs/flowers, and `street` on protest signs, aerial houses, or alleys.

## 5. Evaluation Methodology

The evaluation uses style/semantic queries, object-like queries, and combined queries. Retrieval modes include Qdrant semantic search, keyword-filtered search, and style reranking where applicable. The original automatic metrics use weak heuristic relevance labels based on metadata terms, detected-object payloads, and simple descriptor thresholds.

Keyword-filtered modes achieved high weak-label scores, but these scores should be interpreted carefully. Since the weak evaluator checks metadata terms, and keyword filtering also selects by metadata terms, the evaluation partially rewards the retrieval mode for satisfying its own filter condition. Visual inspection is therefore necessary to determine whether the images actually contain the requested object or scene.

### Visual Inspection Protocol

All PNG result grids in `visualizations/` were inspected manually. Each rank was assigned a visual relevance label: `2` for a full visual match, `1` for a partial match, and `0` for a visual failure. The labels are stored in `visual_inspection.csv`. Visual metrics use the same formulas as the weak-label metrics: Precision@10, average graded relevance, DCG@10, nDCG@10, and MRR@10.

## 6. Automatic Weak-Label Results

Overall automatic metrics:

| mode | query_mode_count | precision_at_10 | avg_relevance | dcg_at_10 | ndcg_at_10 | mrr_at_10 |
| --- | --- | --- | --- | --- | --- | --- |
| qdrant_keyword | 4 | 1.0000 | 2.0000 | 13.6307 | 1.0000 | 1.0000 |
| qdrant_keyword_primary | 4 | 1.0000 | 1.9500 | 13.3080 | 0.9978 | 1.0000 |
| qdrant_keyword_secondary | 3 | 1.0000 | 1.8333 | 12.3045 | 0.9720 | 1.0000 |
| qdrant_rerank | 5 | 1.0000 | 1.8400 | 11.9194 | 0.9442 | 1.0000 |
| qdrant_semantic | 12 | 0.9583 | 1.7250 | 11.4655 | 0.9509 | 1.0000 |

Latency by mode:

| mode | query_mode_count | avg_latency_ms | avg_min_latency_ms | avg_max_latency_ms |
| --- | --- | --- | --- | --- |
| qdrant_keyword | 4 | 511.5591 | 498.8074 | 523.8477 |
| qdrant_keyword_primary | 4 | 522.0376 | 508.6445 | 540.3460 |
| qdrant_keyword_secondary | 3 | 525.3509 | 514.9037 | 533.5395 |
| qdrant_rerank | 5 | 134.3433 | 115.9610 | 150.6085 |
| qdrant_semantic | 12 | 124.7467 | 104.2714 | 139.9344 |

These automatic scores are useful for regression testing, but they are optimistic for keyword-filtered modes because metadata terms influence both retrieval and evaluation.

## 7. Visual Inspection Results

Overall visual metrics:

| mode | query_mode_count | precision_at_10 | avg_relevance | dcg_at_10 | ndcg_at_10 | mrr_at_10 |
| --- | --- | --- | --- | --- | --- | --- |
| qdrant_keyword | 4 | 1.0000 | 1.7750 | 11.6393 | 0.9479 | 1.0000 |
| qdrant_keyword_primary | 4 | 1.0000 | 1.8750 | 12.7462 | 0.9884 | 1.0000 |
| qdrant_keyword_secondary | 3 | 1.0000 | 1.6000 | 10.6708 | 0.9574 | 1.0000 |
| qdrant_rerank | 5 | 1.0000 | 1.7000 | 11.0361 | 0.9352 | 1.0000 |
| qdrant_semantic | 12 | 0.9583 | 1.6750 | 10.9168 | 0.9236 | 1.0000 |

Visual metrics by query group and mode:

| query_group | mode | query_mode_count | precision_at_10 | avg_relevance | dcg_at_10 | ndcg_at_10 | mrr_at_10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| combined | qdrant_keyword_primary | 4 | 1.0000 | 1.8750 | 12.7462 | 0.9884 | 1.0000 |
| combined | qdrant_keyword_secondary | 3 | 1.0000 | 1.6000 | 10.6708 | 0.9574 | 1.0000 |
| combined | qdrant_rerank | 1 | 1.0000 | 2.0000 | 13.6307 | 1.0000 | 1.0000 |
| combined | qdrant_semantic | 4 | 1.0000 | 1.7500 | 11.8825 | 0.9833 | 1.0000 |
| object_like | qdrant_keyword | 4 | 1.0000 | 1.7750 | 11.6393 | 0.9479 | 1.0000 |
| object_like | qdrant_semantic | 4 | 0.8750 | 1.5750 | 10.5667 | 0.9140 | 1.0000 |
| style_semantic | qdrant_rerank | 4 | 1.0000 | 1.6250 | 10.3875 | 0.9190 | 1.0000 |
| style_semantic | qdrant_semantic | 4 | 1.0000 | 1.7000 | 10.3013 | 0.8736 | 1.0000 |

Automatic versus visual metrics:

| query | query_group | mode | auto_precision_at_10 | visual_precision_at_10 | auto_ndcg_at_10 | visual_ndcg_at_10 | precision_delta | ndcg_delta | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| person | object_like | qdrant_semantic | 0.6000 | 0.6000 | 0.8799 | 0.7059 | 0.0000 | -0.1740 | automatic metric overestimates quality |
| person | object_like | qdrant_keyword | 1.0000 | 1.0000 | 1.0000 | 0.8547 | 0.0000 | -0.1453 | keyword metadata likely inflated score |
| minimal architecture | style_semantic | qdrant_semantic | 1.0000 | 1.0000 | 1.0000 | 0.8650 | 0.0000 | -0.1350 | automatic metric agrees with visual inspection |
| dark forest with fog | combined | qdrant_rerank | 1.0000 | 1.0000 | 0.8911 | 1.0000 | 0.0000 | 0.1089 | automatic metric agrees with visual inspection |
| dark forest with fog | combined | qdrant_semantic | 1.0000 | 1.0000 | 0.8911 | 1.0000 | 0.0000 | 0.1089 | automatic metric agrees with visual inspection |
| warm cinematic landscape | style_semantic | qdrant_semantic | 1.0000 | 1.0000 | 0.8178 | 0.7401 | 0.0000 | -0.0778 | automatic metric agrees with visual inspection |
| minimal architecture | style_semantic | qdrant_rerank | 1.0000 | 1.0000 | 1.0000 | 0.9288 | 0.0000 | -0.0712 | style reranking visually improves consistency |
| car | object_like | qdrant_keyword | 1.0000 | 1.0000 | 1.0000 | 0.9409 | 0.0000 | -0.0591 | keyword metadata likely inflated score |
| vibrant summer beach | style_semantic | qdrant_rerank | 1.0000 | 1.0000 | 1.0000 | 0.9412 | 0.0000 | -0.0588 | automatic metric agrees with visual inspection |
| dog on beach | combined | qdrant_keyword_primary | 1.0000 | 1.0000 | 1.0000 | 0.9706 | 0.0000 | -0.0294 | automatic metric agrees with visual inspection |

Representative visual evidence:

![Warm cinematic landscape semantic results](visualizations/warm_cinematic_landscape__qdrant_semantic.png)

**Figure 1.** Semantic retrieval for `warm cinematic landscape`. The grid retrieves landscapes, but many are cool, dark, or only partially warm/cinematic.

![Warm cinematic landscape reranked results](visualizations/warm_cinematic_landscape__qdrant_rerank.png)

**Figure 2.** Reranked retrieval for `warm cinematic landscape`. The reranker improves style consistency by moving warmer, hazier, and more atmospheric landscapes higher in the grid.

![Dog keyword results](visualizations/dog__qdrant_keyword.png)

**Figure 3.** Keyword-filtered retrieval for `dog`. This object keyword performs well in the inspected grid, with mostly clear dog images.

![Person keyword results](visualizations/person__qdrant_keyword.png)

**Figure 4.** Keyword-filtered retrieval for `person`. The filter improves human presence, but several ranks contain small, ambiguous, or non-central people, showing where automatic metadata scores overestimate visual quality.

![Person in street photography keyword-secondary results](visualizations/person_in_street_photography__qdrant_keyword_secondary.png)

**Figure 5.** Keyword-filtered retrieval for `person in street photography` using the `street` keyword. The filter improves urban context, but does not always guarantee a visible central person.

![Car at night keyword-primary results](visualizations/car_at_night__qdrant_keyword_primary.png)

**Figure 6.** Keyword-filtered retrieval for `car at night` using the primary `car` keyword. The filter visibly improves full-query matches by keeping cars central in night scenes.

![Car at night keyword-secondary results](visualizations/car_at_night__qdrant_keyword_secondary.png)

**Figure 7.** Keyword-filtered retrieval for `car at night` using the secondary `night` keyword. The filter preserves night scenes, but some results lack a clear central car.

## 8. Discussion

The automatic metrics are optimistic because weak labels rely on the same metadata fields used by keyword filtering. Visual inspection shows that keyword coverage does not equal object-level correctness. Keyword filters are useful as coarse narrowing mechanisms, especially for broad visual concepts such as `building`, `dog`, and primary `car` intent, but they are not reliable object recognition.

Style reranking can improve global appearance. This is clearest for `warm cinematic landscape`, where reranking moves warmer and more atmospheric images higher in the grid. It is less useful when semantic retrieval already satisfies the style query, as in `dark forest with fog`, and it cannot verify object presence.

YOLO is still necessary because the `detected_objects` payload is empty. An object detector would provide independent object-level evidence, allowing the system to verify whether a person, dog, car, or building is visible and central rather than merely mentioned in metadata.

## 9. Limitations

- Visual labels were assigned by manual inspection of generated grids, not by a multi-annotator relevance study.
- Automatic labels remain weak diagnostics and are not ground truth.
- Unsplash metadata keywords are noisy and over-inclusive.
- Qdrant is running in local path mode; collections above 20k points can trigger local-mode scalability warnings.
- No object detector currently populates `detected_objects`.

## 10. Next Steps

- Add a YOLO object detection pipeline.
- Store detected objects in SQLite and Qdrant payloads.
- Add an object-aware reranker.
- Replace strict keyword filtering with multi-signal scoring.
- Move Qdrant from local mode to Docker/server mode.
- Repeat this evaluation after YOLO is integrated.

## 11. Conclusion

The scaled retrieval pipeline is technically successful and operational at 24,916 images. However, visual inspection reveals that retrieval quality is limited by metadata noise. Keyword filtering is useful as a coarse narrowing mechanism, but it is not reliable object recognition. Style reranking improves global visual consistency for style-sensitive queries, but it does not solve object-specific intent. The next necessary step is to add YOLO object detection, store detected objects in SQLite and Qdrant, and repeat the same evaluation with object-aware retrieval.
