# Validation Set

This stage prepares a structured validation set for evaluating search quality in the final project. It does not run expensive experiments or create generated outputs. Instead, it defines reusable queries and a manual relevance-labeling template that can be applied to search results from the FAISS baseline, style-aware reranking, Qdrant retrieval, object filters, and YOLO object-aware reranking.

The validation set supports the final project requirement to evaluate retrieval quality, not just system functionality or scalability. Once search results are generated and manually labeled, the labels can be used to compute ranking metrics such as Precision@5, Precision@10, nDCG@10, and MRR.

## Files

| File | Purpose |
| --- | --- |
| `queries.csv` | Thirty validation queries grouped by semantic text, style text, object text, mixed text, and image-to-image search. |
| `relevance_labels_template.csv` | Placeholder rows for manual top-10 relevance labels for the first five queries. |
| `query_groups.md` | Explanation of each query group and the systems each group is intended to test. |
| `presentation_summary.md` | Short slide-ready summary for the final presentation. |

## Using `queries.csv`

Run each query against the systems being compared and save the top-k image results. The `intended_system` column indicates the main retrieval modes each query is designed to evaluate.

The query types are:

- `semantic_text`: broad visual and semantic CLIP retrieval.
- `style_text`: style-aware behavior such as warmth, darkness, contrast, saturation, and cinematic mood.
- `object_text`: object retrieval, object filters, and YOLO object-aware reranking.
- `mixed_text`: combined object, style, and scene constraints.
- `image_to_image`: visual similarity to an existing dataset image.

The two `image_to_image` rows are placeholders. Replace them with real local image IDs after sampling from the indexed dataset.

## Filling `relevance_labels_template.csv`

After generating search results, copy or extend the template so each evaluated system has rows for each query and rank. Fill in:

- `image_id`: the returned image identifier.
- `relevance`: the manual relevance judgment.
- `comment`: optional notes about why the result is relevant or not relevant.

Use this relevance scale:

- `2` = highly relevant
- `1` = partially relevant
- `0` = not relevant

## Future Metrics

The completed labels can be used to compute:

- Precision@5: fraction of the top 5 results with relevance greater than 0.
- Precision@10: fraction of the top 10 results with relevance greater than 0.
- nDCG@10: ranking quality that rewards highly relevant results near the top.
- MRR: reciprocal rank of the first relevant result.

## Scope

This validation set is for real-image search quality evaluation. The synthetic 500k corpus is separate and should be used for scalability, indexing, storage, and latency evaluation without claiming real-image relevance.
