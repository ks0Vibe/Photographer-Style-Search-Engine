# Relevance Labeling

This experiment stage prepares a deduplicated manual relevance-judgment pool from the ranked validation search results.

No final search-quality metrics are calculated here. Metrics are intentionally left for the next experiment stage after manual labels are complete.

## Input

- `experiments/09_validation_results/all_validation_results.csv`

The input contains ranked top-k outputs from FAISS, Qdrant, filtering, style reranking, and object-aware reranking systems.

## Outputs

| File | Purpose |
| --- | --- |
| `relevance_judgments.csv` | Primary manual labeling file, one row per unique query-image pair. |
| `relevance_judgments_by_query.csv` | Same judgments sorted for query-by-query review. |
| `unlabeled_items.csv` | Rows where `relevance` is still empty. |
| `labeled_items.csv` | Rows where `relevance` has been completed. |
| `labeling_progress.md` | Regenerated progress summary for reporting. |
| `labeling_guide.md` | Human labeling instructions and relevance rules. |
| `presentation_summary.md` | Short slide-ready summary of the labeling method. |

## Duplicate Handling

The same image can appear for the same query in multiple systems. This stage keeps one judgment row per unique `query_id + image_id` pair and stores the contributing systems in `source_systems`.

Each row also preserves the best rank, ranks by system, highest available score, image path, detected objects, matched keywords, and visual descriptor fields where available.

## Label Preservation

When `relevance_judgments.csv` already exists, the preparation script preserves any non-empty `relevance`, `confidence`, and `comment` values by matching on `judgment_id`.

## Workflow

1. Generate validation search results.
2. Prepare unique relevance judgments.
3. Open `relevance_judgments.csv`.
4. Fill `relevance`, `confidence`, and `comment`.
5. Run `check_relevance_labeling.py`.
6. Proceed to the metrics evaluation stage.

## Run Preparation

```powershell
python experiments/scripts/prepare_relevance_labeling.py
```

Optional:

```powershell
python experiments/scripts/prepare_relevance_labeling.py --input-path experiments/09_validation_results/all_validation_results.csv --output-dir experiments/10_relevance_labeling --max-items-per-query 50
```

## Manual Labeling

Fill only these manual fields:

- `relevance`: `2`, `1`, or `0`
- `confidence`: optional `high`, `medium`, or `low`
- `comment`: optional short note

Use `labeling_guide.md` for the detailed rules. Judge visual relevance to the query, not the CLIP score, keyword text, or detector labels alone.

## Validate Labels

```powershell
python experiments/scripts/check_relevance_labeling.py
```
