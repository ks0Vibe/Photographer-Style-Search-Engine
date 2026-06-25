# Qualitative Visual Findings

## Summary

Visual inspection was completed for all 28 PNG result grids. The inspection confirms that the scaled pipeline is operational, but it also shows that weak automatic labels are optimistic for keyword-filtered modes because they evaluate metadata terms that keyword filtering also uses.

## Cases where keyword filtering helps

Keyword filtering visibly helps when the keyword corresponds to a broad, visually stable concept. `building` returns consistently architectural results, `dog` returns mostly clear dog images, and the primary `car` keyword improves `car at night` by keeping cars central in night scenes.

| query | query_group | mode | precision_at_10 | avg_relevance | ndcg_at_10 |
| --- | --- | --- | --- | --- | --- |
| building | object_like | qdrant_keyword | 1.0000 | 2.0000 | 1.0000 |
| car at night | combined | qdrant_keyword_primary | 1.0000 | 1.9000 | 0.9932 |
| dark forest with fog | combined | qdrant_keyword_primary | 1.0000 | 2.0000 | 1.0000 |
| dog | object_like | qdrant_keyword | 1.0000 | 1.9000 | 0.9960 |
| dog on beach | combined | qdrant_keyword_primary | 1.0000 | 1.7000 | 0.9706 |
| dog on beach | combined | qdrant_keyword_secondary | 1.0000 | 1.6000 | 0.9705 |
| person | object_like | qdrant_keyword | 1.0000 | 1.7000 | 0.8547 |
| person in street photography | combined | qdrant_keyword_primary | 1.0000 | 1.9000 | 0.9897 |

## Cases where keyword filtering fails

Keyword filtering fails or becomes partial when the keyword preserves only one part of the query. For `car at night`, the secondary `night` keyword returns night scenes, gas stations, and roads where the car is not always central. For `person in street photography`, the `street` keyword can preserve urban context while reducing the prominence of the person.

| query | mode | rank | image_id | visual_relevance | failure_reason | visual_notes |
| --- | --- | --- | --- | --- | --- | --- |
| person | qdrant_keyword | 1 | nM5uErhbiGw | 1 | partial_match | Partial visual match. The person keyword improves visible human presence, although some people remain small or ambiguous |
| person | qdrant_keyword | 4 | uMiU1Tmbd1Q | 1 | partial_match | Partial visual match. The person keyword improves visible human presence, although some people remain small or ambiguous |
| person | qdrant_keyword | 7 | PWPjBNarV0k | 1 | partial_match | Partial visual match. The person keyword improves visible human presence, although some people remain small or ambiguous |
| car | qdrant_keyword | 3 | pl1Pp-mDzfw | 1 | partial_match | Partial visual match. Keyword filtering keeps vehicles visible, but centrality varies and some results are partial vehic |
| car | qdrant_keyword | 4 | 82JvJrB9deY | 1 | partial_match | Partial visual match. Keyword filtering keeps vehicles visible, but centrality varies and some results are partial vehic |
| car | qdrant_keyword | 6 | fsBOigltAQc | 1 | partial_match | Partial visual match. Keyword filtering keeps vehicles visible, but centrality varies and some results are partial vehic |
| car | qdrant_keyword | 8 | hPbMO39Qcec | 1 | partial_match | Partial visual match. Keyword filtering keeps vehicles visible, but centrality varies and some results are partial vehic |
| car | qdrant_keyword | 9 | OSbw1oA6Mjo | 1 | partial_match | Partial visual match. Keyword filtering keeps vehicles visible, but centrality varies and some results are partial vehic |
| dog | qdrant_keyword | 8 | yCKKd37OsgI | 1 | partial_match | Partial visual match. The dog keyword grid is visually almost identical to semantic retrieval and mostly contains clear  |
| person in street photography | qdrant_keyword_primary | 6 | H5llhoZSm9Y | 1 | partial_match | Partial visual match. The person keyword preserves visible people but does not improve the already strong street-photo c |

## Cases where style reranking helps

For `warm cinematic landscape`, reranking visibly improves global appearance by moving warmer, hazier, and more atmospheric landscapes higher in the grid.

| query | query_group | mode | auto_ndcg_at_10 | visual_ndcg_at_10 | interpretation |
| --- | --- | --- | --- | --- | --- |
| minimal architecture | style_semantic | qdrant_rerank | 1.0000 | 0.9288 | style reranking visually improves consistency |
| warm cinematic landscape | style_semantic | qdrant_rerank | 0.9387 | 0.9296 | style reranking visually improves consistency |

## Cases where style reranking does not help

Reranking does not always improve visual relevance. `dark forest with fog` was already very strong under semantic retrieval, so reranking mostly preserved quality. For beach and minimal architecture, reranking changed style emphasis but did not clearly improve every rank.

| query | query_group | mode | auto_ndcg_at_10 | visual_ndcg_at_10 | interpretation |
| --- | --- | --- | --- | --- | --- |
| dark forest with fog | combined | qdrant_rerank | 0.8911 | 1.0000 | automatic metric agrees with visual inspection |
| dark moody forest | style_semantic | qdrant_rerank | 0.8911 | 0.8767 | automatic metric agrees with visual inspection |
| vibrant summer beach | style_semantic | qdrant_rerank | 1.0000 | 0.9412 | automatic metric agrees with visual inspection |

## Evidence that YOLO is needed

The grids show that metadata keyword coverage is not the same as object-level correctness. Keyword filters can select payloads containing `person`, `car`, `dog`, or `street`, but they cannot verify whether the object is visually central. The `detected_objects` payload is currently empty, so the system has no independent object evidence. YOLO is needed to populate detected objects in SQLite and Qdrant and to support object-aware retrieval and reranking.

## Automatic vs visual metric mismatch

The largest mismatches occur where weak automatic labels reward metadata agreement more than visual correctness. These cases should be treated as diagnostics rather than final quality claims.

| query | query_group | mode | auto_precision_at_10 | visual_precision_at_10 | auto_ndcg_at_10 | visual_ndcg_at_10 | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| car | object_like | qdrant_keyword | 1.0000 | 1.0000 | 1.0000 | 0.9409 | keyword metadata likely inflated score |
| person | object_like | qdrant_keyword | 1.0000 | 1.0000 | 1.0000 | 0.8547 | keyword metadata likely inflated score |
| person | object_like | qdrant_semantic | 0.6000 | 0.6000 | 0.8799 | 0.7059 | automatic metric overestimates quality |
