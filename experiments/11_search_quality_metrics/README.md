# Search Quality Metrics

This stage evaluates retrieval systems against the shared human relevance judgments from `experiments/10_relevance_labeling/relevance_judgments.csv`.

The same query-image judgment is reused across every system, so duplicate results from different retrieval approaches are judged once and compared fairly.

## Inputs

- `experiments/09_validation_results/all_validation_results.csv`
- `experiments/10_relevance_labeling/relevance_judgments.csv`

## Metrics

- Precision@k: relevant results in the available top-k divided by the number of available retrieved results.
- DCG@k: `sum((2^rel_i - 1) / log2(i + 1))`, with ranks starting at 1.
- nDCG@k: DCG divided by ideal DCG; returns 0 when ideal DCG is 0.
- MRR: reciprocal rank of the first result with relevance at or above the threshold.
- Success@k: 1 if at least one relevant result appears in top-k, otherwise 0.
- Mean relevance@10: average graded relevance over available top-10 results.

## Relevance

The graded labels are:

- `2`: highly relevant
- `1`: partially relevant
- `0`: not relevant

Binary metrics use `relevance >= 1` by default. Use `--relevance-threshold 2` for stricter binary metrics.

## Macro Averaging

`metrics_by_system.csv` reports macro averages: each query contributes one query-system score, and system scores are averaged across queries. This avoids letting queries with more retrieved rows dominate the main comparison.

## Run

```powershell
.\.venv\Scripts\python.exe experiments\scripts\evaluate_search_quality.py
```

For an incomplete-label test run:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\evaluate_search_quality.py --allow-incomplete-labels
```

For strict relevance:

```powershell
.\.venv\Scripts\python.exe experiments\scripts\evaluate_search_quality.py --relevance-threshold 2
```

## Outputs

| File | Purpose |
| --- | --- |
| `metrics_by_query.csv` | One metric row per query and system. |
| `metrics_by_system.csv` | Macro-averaged comparison by system. |
| `metrics_by_query_type.csv` | Macro-averaged comparison by query type and system. |
| `metrics_statistical_summary.csv` | Mean, standard deviation, minimum, and maximum by system and metric. |
| `metrics_confidence_intervals.csv` | Bootstrap 95% confidence intervals by system and metric. |
| `system_comparison.md` | Presentation-friendly system comparison. |
| `failure_analysis.md` | Automatically identified weak queries, system gaps, and regressions. |
| `presentation_summary.md` | Slide-ready summary. |
| `evaluation_summary.md` | Run metadata, warnings, and generated outputs. |

## Limitations

Incomplete labels are never treated as irrelevant by default. Strict runs stop when needed labels are missing. With `--allow-incomplete-labels`, metrics are based only on labeled result rows and reports are marked preliminary.

This stage contributes the final comparative search-quality evidence for the report and presentation, but it does not alter rankings or manual labels.
