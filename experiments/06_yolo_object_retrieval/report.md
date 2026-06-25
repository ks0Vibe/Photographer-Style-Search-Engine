# YOLO Object Retrieval Evaluation

## 1. Goal

This experiment tests whether adding YOLO detected objects improves object-specific retrieval after the previous 25k-scale evaluation found `object_coverage = 0.0` and showed that object search relied on noisy metadata keywords.

The main goal is to check whether the system can move from metadata-only object search to retrieval based on independent visual object evidence.

## 2. Implementation

YOLO detections are extracted with `ultralytics` using `yolov8n.pt` by default. Detected object labels are stored as normalized JSON lists in `images.detected_objects`, with `detection_model` and `detection_updated_at` metadata. Qdrant payload upload now reads those SQLite values and stores them in the `detected_objects` payload field.

Before YOLO, object search relied only on noisy metadata keywords. After YOLO, object filters can use independent visual evidence from detected objects. The full CPU YOLO pass was completed over the local corpus: 24,873 images were processed, 43 already-existing rows were skipped, and no image files were missing.

## 3. Object Payload Statistics

| stat                         |  value |
| ---------------------------- | -----: |
| qdrant_points                |  24916 |
| images_with_detected_objects |  11418 |
| object_coverage              | 0.4583 |
| unique_detected_objects      |     78 |

Top detected object classes:

| object       | count |
| ------------ | ----: |
| person       |  3923 |
| bird         |  1489 |
| dog          |  1104 |
| potted plant |   716 |
| cat          |   594 |
| vase         |   558 |
| bear         |   405 |
| boat         |   345 |
| horse        |   335 |
| car          |   303 |
| kite         |   265 |
| cow          |   246 |
| sheep        |   243 |
| umbrella     |   243 |
| bed          |   234 |
| frisbee      |   188 |
| elephant     |   179 |
| surfboard    |   169 |
| sports ball  |   153 |
| orange       |   152 |

After YOLO, 11,418 out of 24,916 Qdrant points have at least one detected object, giving an object coverage of 45.83%. This does not mean that the detector failed on the remaining images. Many Unsplash images are landscapes, skies, forests, architecture, textures, abstract photos, or other scenes without COCO-detectable foreground objects. In addition, YOLOv8n uses a fixed COCO class set, so non-COCO concepts such as `building`, `forest`, `street photography`, or `cinematic landscape` cannot be detected as strict object classes.

## 4. Retrieval Modes

* `qdrant_semantic`: CLIP/Qdrant text retrieval without payload filters.
* `qdrant_keyword`: keyword payload filter using the requested object label.
* `qdrant_object`: detected-object payload filter.
* `qdrant_keyword_object`: both keyword and detected-object filters.
* `qdrant_object_rerank`: semantic candidate pool reranked by semantic, object, keyword, and optional style scores.

The important difference between `qdrant_keyword` and `qdrant_object` is that keyword filtering uses metadata labels, while object filtering uses YOLO visual detections. Therefore, object filtering is less dependent on noisy Unsplash keyword annotations.

## 5. Evaluation Methodology

The query set contains object-like and combined object/context queries. Weak labels use detected objects plus metadata context. These labels are more reliable than pre-YOLO keyword-only labels because `detected_objects` is independent from Unsplash metadata keywords, but they are still not human ground truth.

Visual grids were generated and `visual_inspection_template.csv` is provided for manual labels. Final human visual inspection is still pending, so the reported metrics should be interpreted as automatic diagnostic metrics rather than final relevance judgments.

## 6. Results

Overall metrics:

| mode                  | query_mode_count | precision_at_10 | avg_relevance | dcg_at_10 | ndcg_at_10 | mrr_at_10 | latency_ms |
| --------------------- | ---------------: | --------------: | ------------: | --------: | ---------: | --------: | ---------: |
| qdrant_keyword        |               12 |          1.0000 |        1.6417 |   10.4937 |     0.9599 |    1.0000 |   523.2914 |
| qdrant_keyword_object |               10 |          1.0000 |        1.9100 |   12.8105 |     0.9754 |    1.0000 |   514.5898 |
| qdrant_object         |               10 |          1.0000 |        1.9100 |   12.8105 |     0.9754 |    1.0000 |   325.7668 |
| qdrant_object_rerank  |               12 |          1.0000 |        1.7583 |   11.4327 |     0.9795 |    1.0000 |   120.0051 |
| qdrant_semantic       |               12 |          0.9583 |        1.5167 |    9.7142 |     0.9518 |    1.0000 |   107.7954 |

Query group metrics:

| query_group | mode                  | query_mode_count | precision_at_10 | avg_relevance | dcg_at_10 | ndcg_at_10 | mrr_at_10 | latency_ms |
| ----------- | --------------------- | ---------------: | --------------: | ------------: | --------: | ---------: | --------: | ---------: |
| combined    | qdrant_keyword        |                6 |          1.0000 |        1.6167 |   10.1700 |     0.9446 |    1.0000 |   544.0774 |
| combined    | qdrant_keyword_object |                5 |          1.0000 |        1.8200 |   11.9903 |     0.9508 |    1.0000 |   515.2688 |
| combined    | qdrant_object         |                5 |          1.0000 |        1.8200 |   11.9903 |     0.9508 |    1.0000 |   324.9735 |
| combined    | qdrant_object_rerank  |                6 |          1.0000 |        1.6833 |   10.7492 |     0.9590 |    1.0000 |   111.1921 |
| combined    | qdrant_semantic       |                6 |          1.0000 |        1.5333 |    9.6159 |     0.9484 |    1.0000 |   107.1429 |
| object_like | qdrant_keyword        |                6 |          1.0000 |        1.6667 |   10.8173 |     0.9753 |    1.0000 |   502.5054 |
| object_like | qdrant_keyword_object |                5 |          1.0000 |        2.0000 |   13.6307 |     1.0000 |    1.0000 |   513.9109 |
| object_like | qdrant_object         |                5 |          1.0000 |        2.0000 |   13.6307 |     1.0000 |    1.0000 |   326.5601 |
| object_like | qdrant_object_rerank  |                6 |          1.0000 |        1.8333 |   12.1162 |     1.0000 |    1.0000 |   128.8180 |
| object_like | qdrant_semantic       |                6 |          0.9167 |        1.5000 |    9.8125 |     0.9552 |    1.0000 |   108.4480 |

Strict object filtering improved object-query reliability for COCO-supported classes detected by YOLO, especially `person`, `bird`, `dog`, `cat`, and `car`. The `qdrant_object` mode achieved higher average relevance than keyword filtering while also being faster than keyword-only and keyword+object filtering in this local Qdrant setup.

Object-aware reranking achieved the highest overall nDCG@10 among the evaluated modes. This suggests that reranking can be a less brittle alternative to strict filtering: instead of excluding all candidates without a matching object payload, it keeps semantic candidates and promotes images with object evidence.

Best examples:

| query            | query_group | mode                  | precision_at_10 | avg_relevance | ndcg_at_10 | latency_ms |
| ---------------- | ----------- | --------------------- | --------------: | ------------: | ---------: | ---------: |
| bird             | object_like | qdrant_keyword_object |          1.0000 |        2.0000 |     1.0000 |   512.6672 |
| bird             | object_like | qdrant_object         |          1.0000 |        2.0000 |     1.0000 |   344.6644 |
| bird             | object_like | qdrant_object_rerank  |          1.0000 |        2.0000 |     1.0000 |   140.7982 |
| building         | object_like | qdrant_keyword        |          1.0000 |        1.0000 |     1.0000 |   512.9372 |
| building         | object_like | qdrant_object_rerank  |          1.0000 |        1.0000 |     1.0000 |   131.5969 |
| building         | object_like | qdrant_semantic       |          1.0000 |        1.0000 |     1.0000 |   111.4140 |
| building in city | combined    | qdrant_keyword        |          1.0000 |        1.0000 |     1.0000 |   641.5880 |
| building in city | combined    | qdrant_object_rerank  |          1.0000 |        1.0000 |     1.0000 |   119.2171 |

The `building` examples should be interpreted carefully. `Building` is not a COCO object class in this YOLO setup, so strict object-filtered modes cannot directly support it. For this type of query, CLIP semantic search and metadata keywords remain necessary.

Worst examples:

| query       | query_group | mode                  | precision_at_10 | avg_relevance | ndcg_at_10 | latency_ms |
| ----------- | ----------- | --------------------- | --------------: | ------------: | ---------: | ---------: |
| cat indoors | combined    | qdrant_keyword        |          1.0000 |        1.5000 |     0.7947 |   503.1224 |
| cat indoors | combined    | qdrant_keyword_object |          1.0000 |        1.5000 |     0.7947 |   486.1825 |
| cat indoors | combined    | qdrant_object         |          1.0000 |        1.5000 |     0.7947 |   347.3628 |
| cat indoors | combined    | qdrant_object_rerank  |          1.0000 |        1.5000 |     0.7947 |    87.6752 |
| cat indoors | combined    | qdrant_semantic       |          1.0000 |        1.5000 |     0.7947 |   102.4820 |
| person      | object_like | qdrant_semantic       |          0.6000 |        1.1000 |     0.8639 |   141.5558 |
| car         | object_like | qdrant_semantic       |          0.9000 |        1.2000 |     0.8982 |    81.4371 |
| car         | object_like | qdrant_keyword        |          1.0000 |        1.5000 |     0.9038 |   508.0634 |

The lower-scoring cases show that object detection alone does not fully solve combined intent. For example, `cat indoors` requires both a visible cat and an indoor context. YOLO can help verify the cat, but it does not directly verify all scene-level context. This is why object-aware reranking and semantic CLIP evidence remain important.

## 7. Visual Findings

Visualization grids were generated, but final visual relevance judgments require manual inspection. Use `visual_inspection_template.csv` to record visual labels.

![Object filter for person.](visualizations/person__qdrant_object.png)

**Figure 1.** Object filter for `person`. This grid should be inspected to verify whether detected people are visible and central, rather than merely present somewhere in the image.

![Object filter for car at night.](visualizations/car_at_night__qdrant_object.png)

**Figure 2.** Object filter for `car at night`. This grid tests whether object filtering retrieves cars while preserving the night context.

![Object-aware rerank for dog on beach.](visualizations/dog_on_beach__qdrant_object_rerank.png)

**Figure 3.** Object-aware rerank for `dog on beach`. This grid tests whether reranking can balance object evidence with scene context.

![Keyword + object filter for bird in nature.](visualizations/bird_in_nature__qdrant_keyword_object.png)

**Figure 4.** Keyword + object filter for `bird in nature`. This grid tests whether combining metadata and detected-object evidence improves object-specific retrieval.

## 8. Discussion

YOLO improves object-specific retrieval for images where the requested concept belongs to the supported COCO object classes and is detected with sufficient confidence. This is visible in the aggregate metrics: `qdrant_object` achieves higher average relevance than `qdrant_keyword`, and `qdrant_object_rerank` achieves the strongest overall nDCG@10.

Keyword filtering can still be useful, but it remains limited by metadata noise. A keyword such as `dog`, `person`, or `car` may be present in the payload even when the object is small, ambiguous, or not visually central. YOLO detections provide cleaner object-level evidence because they come from the image itself rather than from external metadata.

Strict object filters can also be too narrow. If YOLO misses a small, occluded, unusual, or unsupported object, strict filtering can exclude otherwise relevant CLIP candidates. Object-aware reranking is less brittle because it keeps the semantic candidate pool and uses object evidence as a ranking signal rather than a hard requirement.

### Why object coverage is not 100%

The object coverage after YOLO is 45.83%. This does not mean that the detection pipeline failed on the remaining images. Instead, many Unsplash images are landscapes, abstract photos, architecture, skies, forests, close-up textures, or other scenes without COCO-detectable objects. In addition, YOLOv8n is trained on a fixed COCO label set, so non-COCO concepts such as `building`, `forest`, `street photography`, or `cinematic landscape` cannot be detected as strict object classes.

Therefore, YOLO improves object-specific retrieval, but it complements rather than replaces CLIP semantic retrieval, metadata keywords, and style descriptors.

### COCO class limitation: `building`

The `building` query demonstrates a limitation of COCO-based YOLO detection. Since `building` is not a supported YOLOv8n/COCO class in this setup, strict object-filtered modes cannot cover this query directly. In this case, CLIP semantic retrieval and metadata keywords remain necessary. For broader scene and architecture concepts, future work should consider open-vocabulary detectors such as GroundingDINO or OWL-ViT.

## 9. Limitations

* YOLO may miss small, occluded, unusual, or low-confidence objects.
* YOLOv8n uses a fixed COCO class set and is not open vocabulary.
* Scene-level concepts such as `building`, `forest`, `street photography`, and `cinematic landscape` are not handled as strict object classes.
* No multi-annotator visual labels are included yet.
* The current visual inspection file is still a template and requires manual completion.
* Qdrant local mode is a scalability limitation for collections above 20k points.
* CPU inference is slow for the full 24,916-image corpus.

## 10. Next Steps

* Complete manual visual inspection using `visual_inspection_template.csv`.
* Tune the YOLO confidence threshold.
* Store bounding boxes and confidence scores, not only object labels.
* Add object-aware reranking to the main search service.
* Use Docker/server Qdrant instead of local path mode.
* Consider GroundingDINO or OWL-ViT for open-vocabulary object and scene concepts.
* Repeat the evaluation with human visual labels.

## 11. Conclusion

YOLO directly addresses the main limitation found in the previous scaled retrieval evaluation: before this stage, `detected_objects` coverage was 0%, so object-specific search depended only on noisy metadata keywords. After running full-corpus YOLO extraction and rebuilding Qdrant payloads, 11,418 out of 24,916 images contain at least one detected object, giving an object coverage of 45.83% across 78 detected classes.

The new object payloads make retrieval more reliable for COCO-supported object queries such as `person`, `dog`, `cat`, `bird`, and `car`. Strict object filtering achieved higher weak-label relevance than keyword filtering, showing that YOLO detections provide cleaner object-level evidence than metadata keywords. Object-aware reranking is also useful because it is less brittle than strict filtering: it can preserve semantic candidates while promoting images with matching detected objects.

However, YOLO does not solve all visual search cases. Many photographer-style queries involve scenes, styles, and open-vocabulary concepts such as `building`, `street photography`, `minimal architecture`, `forest`, or `cinematic landscape`, which are not always covered by COCO object classes. For these cases, CLIP semantic search, metadata keywords, style descriptors, and future open-vocabulary detectors remain important.

Overall, the project has progressed from CLIP-only semantic retrieval to a multi-signal image search system combining CLIP embeddings, Qdrant payload filters, metadata keywords, visual style descriptors, YOLO object detections, and object-aware reranking.
