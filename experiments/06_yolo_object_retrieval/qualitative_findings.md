# YOLO Object Retrieval Qualitative Findings

Visualization grids were generated for manual review. Final visual findings should be filled after inspecting `visual_inspection_template.csv` and the PNG grids.

## Payload Coverage

- Images with detected objects in Qdrant: 11418
- Object coverage: 0.4583

## Object Filter Diagnostics

| query | query_group | mode | precision_at_10 | avg_relevance | ndcg_at_10 |
| --- | --- | --- | --- | --- | --- |
| bird | object_like | qdrant_object | 1.0000 | 2.0000 | 1.0000 |
| bird in nature | combined | qdrant_object | 1.0000 | 1.8000 | 0.9680 |
| car | object_like | qdrant_object | 1.0000 | 2.0000 | 1.0000 |
| car at night | combined | qdrant_object | 1.0000 | 2.0000 | 1.0000 |
| cat | object_like | qdrant_object | 1.0000 | 2.0000 | 1.0000 |
| cat indoors | combined | qdrant_object | 1.0000 | 1.5000 | 0.7947 |
| dog | object_like | qdrant_object | 1.0000 | 2.0000 | 1.0000 |
| dog on beach | combined | qdrant_object | 1.0000 | 2.0000 | 1.0000 |
| person | object_like | qdrant_object | 1.0000 | 2.0000 | 1.0000 |
| person in street photography | combined | qdrant_object | 1.0000 | 1.8000 | 0.9911 |

## Object-Aware Rerank Diagnostics

| query | query_group | mode | precision_at_10 | avg_relevance | ndcg_at_10 |
| --- | --- | --- | --- | --- | --- |
| bird | object_like | qdrant_object_rerank | 1.0000 | 2.0000 | 1.0000 |
| bird in nature | combined | qdrant_object_rerank | 1.0000 | 1.8000 | 0.9680 |
| building | object_like | qdrant_object_rerank | 1.0000 | 1.0000 | 1.0000 |
| building in city | combined | qdrant_object_rerank | 1.0000 | 1.0000 | 1.0000 |
| car | object_like | qdrant_object_rerank | 1.0000 | 2.0000 | 1.0000 |
| car at night | combined | qdrant_object_rerank | 1.0000 | 2.0000 | 1.0000 |
| cat | object_like | qdrant_object_rerank | 1.0000 | 2.0000 | 1.0000 |
| cat indoors | combined | qdrant_object_rerank | 1.0000 | 1.5000 | 0.7947 |
| dog | object_like | qdrant_object_rerank | 1.0000 | 2.0000 | 1.0000 |
| dog on beach | combined | qdrant_object_rerank | 1.0000 | 2.0000 | 1.0000 |
| person | object_like | qdrant_object_rerank | 1.0000 | 2.0000 | 1.0000 |
| person in street photography | combined | qdrant_object_rerank | 1.0000 | 1.8000 | 0.9911 |

## Notes

Keyword noise can still appear in `qdrant_keyword` results. Object filters depend on YOLO recall and will miss images where YOLO did not detect the requested class.
