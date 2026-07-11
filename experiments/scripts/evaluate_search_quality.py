from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "experiments" / "09_validation_results" / "all_validation_results.csv"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "experiments" / "10_relevance_labeling" / "relevance_judgments.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "11_search_quality_metrics"

REQUIRED_RESULT_COLUMNS = ["query_id", "system", "rank", "image_id"]
OPTIONAL_RESULT_COLUMNS = [
    "query",
    "type",
    "score",
    "image_path",
    "matched_keywords",
    "detected_objects",
    "brightness",
    "contrast",
    "saturation",
    "warmth",
    "notes",
]
REQUIRED_JUDGMENT_COLUMNS = ["query_id", "image_id", "relevance"]
OPTIONAL_JUDGMENT_COLUMNS = ["query", "type", "confidence", "comment"]
KNOWN_QUERY_TYPES = {"semantic_text", "style_text", "object_text", "mixed_text", "image_to_image"}
METRIC_COLUMNS = [
    "precision_at_5",
    "precision_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "mrr",
    "success_at_1",
    "success_at_5",
    "mean_relevance_at_10",
]
QUERY_METRIC_COLUMNS = [
    "query_id",
    "query",
    "type",
    "system",
    "result_count",
    "labeled_result_count",
    *METRIC_COLUMNS,
]
SYSTEM_METRIC_COLUMNS = ["system", "query_count", *METRIC_COLUMNS]
QUERY_TYPE_METRIC_COLUMNS = ["type", "system", "query_count", *METRIC_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval systems against shared manual relevance judgments."
    )
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--relevance-threshold", type=int, choices=[1, 2], default=1)
    parser.add_argument("--allow-incomplete-labels", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Validation results CSV not found: {path}")
    results = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_RESULT_COLUMNS if column not in results.columns]
    if missing:
        raise ValueError(f"Validation results are missing required columns: {missing}")
    for column in OPTIONAL_RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = ""
    for column in REQUIRED_RESULT_COLUMNS + OPTIONAL_RESULT_COLUMNS:
        results[column] = results[column].fillna("").astype(str).str.strip()
    results["rank_number"] = pd.to_numeric(results["rank"], errors="coerce")
    results["score_number"] = pd.to_numeric(results["score"], errors="coerce")
    return results


def load_judgments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Relevance judgments CSV not found: {path}")
    judgments = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_JUDGMENT_COLUMNS if column not in judgments.columns]
    if missing:
        raise ValueError(f"Relevance judgments are missing required columns: {missing}")
    for column in OPTIONAL_JUDGMENT_COLUMNS:
        if column not in judgments.columns:
            judgments[column] = ""
    for column in REQUIRED_JUDGMENT_COLUMNS + OPTIONAL_JUDGMENT_COLUMNS:
        judgments[column] = judgments[column].fillna("").astype(str).str.strip()
    return judgments


def validate_relevance_labels(judgments: pd.DataFrame) -> list[str]:
    labels = judgments["relevance"].astype(str).str.strip()
    invalid = judgments[(labels != "") & (~labels.isin({"0", "1", "2"}))]
    if invalid.empty:
        return []
    examples = invalid[["query_id", "image_id", "relevance"]].head(10).to_dict("records")
    return [f"Invalid relevance values found. Allowed values are 0, 1, and 2. Examples: {examples}"]


def validate_result_ranks(results: pd.DataFrame) -> dict[str, Any]:
    duplicate_rows = results[
        results.duplicated(subset=["query_id", "system", "image_id"], keep=False)
    ][["query_id", "system", "image_id", "rank"]]

    duplicate_ranks: list[dict[str, Any]] = []
    rank_gaps: list[dict[str, Any]] = []
    for (query_id, system), group in results.groupby(["query_id", "system"]):
        ranks = pd.to_numeric(group["rank"], errors="coerce").dropna().astype(int).tolist()
        duplicate_rank_values = sorted(rank for rank in set(ranks) if ranks.count(rank) > 1)
        if duplicate_rank_values:
            duplicate_ranks.append(
                {"query_id": query_id, "system": system, "duplicate_ranks": ",".join(map(str, duplicate_rank_values))}
            )
        if ranks:
            expected = set(range(min(ranks), max(ranks) + 1))
            missing = sorted(expected.difference(ranks))
            if missing:
                rank_gaps.append(
                    {"query_id": query_id, "system": system, "missing_ranks": ",".join(map(str, missing[:20]))}
                )

    return {
        "duplicate_system_query_image_rows": duplicate_rows.reset_index(drop=True),
        "duplicate_ranks": pd.DataFrame(duplicate_ranks),
        "rank_gaps": pd.DataFrame(rank_gaps),
    }


def join_results_with_labels(results: pd.DataFrame, judgments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    judgment_subset = judgments[
        ["query_id", "image_id", "relevance", "confidence", "comment", "query", "type"]
    ].rename(columns={"query": "judgment_query", "type": "judgment_type"})

    joined = results.merge(
        judgment_subset,
        on=["query_id", "image_id"],
        how="left",
        indicator=True,
        validate="many_to_one",
    )
    joined["has_judgment"] = joined["_merge"] == "both"
    joined["relevance_text"] = joined["relevance"].fillna("").astype(str).str.strip()
    joined["relevance_value"] = pd.to_numeric(joined["relevance_text"], errors="coerce")
    joined["confidence"] = joined["confidence"].fillna("").astype(str).str.strip()
    joined["comment"] = joined["comment"].fillna("").astype(str).str.strip()
    joined["query"] = joined["query"].where(joined["query"].astype(str).str.strip() != "", joined["judgment_query"])
    joined["type"] = joined["type"].where(joined["type"].astype(str).str.strip() != "", joined["judgment_type"])

    result_keys = set(zip(results["query_id"], results["image_id"], strict=False))
    judgment_keys = set(zip(judgments["query_id"], judgments["image_id"], strict=False))
    unused_judgments = judgments[
        ~judgments.apply(lambda row: (row["query_id"], row["image_id"]) in result_keys, axis=1)
    ][["query_id", "image_id", "relevance", "comment"]]

    data_quality = {
        "result_rows_without_judgments": joined[~joined["has_judgment"]][
            ["query_id", "system", "rank", "image_id"]
        ].reset_index(drop=True),
        "judgments_not_in_results": unused_judgments.reset_index(drop=True),
        "missing_labels": joined[joined["relevance_text"] == ""][
            ["query_id", "query", "type", "system", "rank", "image_id"]
        ].reset_index(drop=True),
    }
    return joined.drop(columns=["_merge", "judgment_query", "judgment_type"]), data_quality


def precision_at_k(relevances: list[int | float], k: int, relevance_threshold: int = 1) -> float:
    top = list(relevances[:k])
    if not top:
        return 0.0
    relevant = sum(1 for value in top if value >= relevance_threshold)
    return relevant / len(top)


def dcg_at_k(relevances: list[int | float], k: int) -> float:
    total = 0.0
    for index, relevance in enumerate(relevances[:k], start=1):
        total += (math.pow(2.0, float(relevance)) - 1.0) / math.log2(index + 1)
    return total


def ndcg_at_k(relevances: list[int | float], k: int) -> float:
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def reciprocal_rank(relevances: list[int | float], relevance_threshold: int = 1) -> float:
    for index, relevance in enumerate(relevances, start=1):
        if relevance >= relevance_threshold:
            return 1.0 / index
    return 0.0


def success_at_k(relevances: list[int | float], k: int, relevance_threshold: int = 1) -> float:
    return 1.0 if any(value >= relevance_threshold for value in relevances[:k]) else 0.0


def mean_relevance_at_k(relevances: list[int | float], k: int) -> float:
    top = list(relevances[:k])
    if not top:
        return 0.0
    return float(np.mean(top))


def evaluate_query_system(
    group: pd.DataFrame,
    *,
    relevance_threshold: int,
    allow_incomplete_labels: bool,
) -> dict[str, Any]:
    ordered = group.sort_values(["rank_number", "rank"], na_position="last", kind="stable")
    labeled = ordered[ordered["relevance_text"] != ""] if allow_incomplete_labels else ordered
    relevances = [int(value) for value in labeled["relevance_value"].dropna().tolist()]
    first = ordered.iloc[0]
    return {
        "query_id": first["query_id"],
        "query": first["query"],
        "type": first["type"],
        "system": first["system"],
        "result_count": len(ordered),
        "labeled_result_count": len(relevances),
        "precision_at_5": precision_at_k(relevances, 5, relevance_threshold),
        "precision_at_10": precision_at_k(relevances, 10, relevance_threshold),
        "ndcg_at_5": ndcg_at_k(relevances, 5),
        "ndcg_at_10": ndcg_at_k(relevances, 10),
        "mrr": reciprocal_rank(relevances, relevance_threshold),
        "success_at_1": success_at_k(relevances, 1, relevance_threshold),
        "success_at_5": success_at_k(relevances, 5, relevance_threshold),
        "mean_relevance_at_10": mean_relevance_at_k(relevances, 10),
    }


def evaluate_all_queries(
    joined: pd.DataFrame,
    *,
    relevance_threshold: int,
    allow_incomplete_labels: bool,
) -> pd.DataFrame:
    rows = [
        evaluate_query_system(
            group,
            relevance_threshold=relevance_threshold,
            allow_incomplete_labels=allow_incomplete_labels,
        )
        for _, group in joined.groupby(["query_id", "system"], sort=True)
    ]
    metrics = pd.DataFrame(rows, columns=QUERY_METRIC_COLUMNS)
    if metrics.empty:
        return metrics
    metrics = metrics.sort_values(["query_id", "system"], kind="stable").reset_index(drop=True)
    return metrics


def aggregate_by_system(metrics_by_query: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system, group in metrics_by_query.groupby("system", sort=False):
        row = {"system": system, "query_count": int(group["query_id"].nunique())}
        row.update({metric: float(group[metric].mean()) if not group.empty else 0.0 for metric in METRIC_COLUMNS})
        rows.append(row)
    metrics = pd.DataFrame(rows, columns=SYSTEM_METRIC_COLUMNS)
    if metrics.empty:
        return metrics
    return metrics.sort_values("ndcg_at_10", ascending=False, kind="stable").reset_index(drop=True)


def aggregate_by_query_type(metrics_by_query: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (query_type, system), group in metrics_by_query.groupby(["type", "system"], sort=True):
        row = {"type": query_type, "system": system, "query_count": int(group["query_id"].nunique())}
        row.update({metric: float(group[metric].mean()) if not group.empty else 0.0 for metric in METRIC_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows, columns=QUERY_TYPE_METRIC_COLUMNS)


def build_statistical_summary(metrics_by_query: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system, group in metrics_by_query.groupby("system", sort=True):
        for metric in METRIC_COLUMNS:
            values = group[metric].astype(float)
            rows.append(
                {
                    "system": system,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else 0.0,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(values.min()) if len(values) else 0.0,
                    "max": float(values.max()) if len(values) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_confidence_intervals(
    metrics_by_query: pd.DataFrame,
    *,
    samples: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if samples <= 0:
        return pd.DataFrame(columns=["system", "metric", "mean", "ci95_low", "ci95_high", "bootstrap_samples"])
    rng = np.random.default_rng(random_seed)
    for system, group in metrics_by_query.groupby("system", sort=True):
        for metric in METRIC_COLUMNS:
            values = group[metric].astype(float).to_numpy()
            if len(values) == 0:
                low = high = mean = 0.0
            else:
                sample_means = [
                    float(np.mean(rng.choice(values, size=len(values), replace=True)))
                    for _ in range(samples)
                ]
                low, high = np.percentile(sample_means, [2.5, 97.5])
                mean = float(np.mean(values))
            rows.append(
                {
                    "system": system,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "bootstrap_samples": samples,
                }
            )
    return pd.DataFrame(rows)


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def write_csv_outputs(
    output_dir: Path,
    metrics_by_query: pd.DataFrame,
    metrics_by_system: pd.DataFrame,
    metrics_by_query_type: pd.DataFrame,
    statistical_summary: pd.DataFrame,
    confidence_intervals: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_query.to_csv(output_dir / "metrics_by_query.csv", index=False)
    metrics_by_system.to_csv(output_dir / "metrics_by_system.csv", index=False)
    metrics_by_query_type.to_csv(output_dir / "metrics_by_query_type.csv", index=False)
    statistical_summary.to_csv(output_dir / "metrics_statistical_summary.csv", index=False)
    confidence_intervals.to_csv(output_dir / "metrics_confidence_intervals.csv", index=False)


def best_system(metrics: pd.DataFrame, metric: str) -> str:
    if metrics.empty or metric not in metrics.columns:
        return "n/a"
    best_value = float(metrics[metric].max())
    tied = metrics[metrics[metric].astype(float).map(lambda value: math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12))]
    systems = [str(system) for system in tied["system"].tolist()]
    if len(systems) == 1:
        return systems[0]
    return "tie: " + ", ".join(systems)


def build_main_comparison_table(metrics_by_system: pd.DataFrame) -> list[str]:
    columns = [
        ("system", "System"),
        ("precision_at_5", "P@5"),
        ("precision_at_10", "P@10"),
        ("ndcg_at_10", "nDCG@10"),
        ("mrr", "MRR"),
        ("success_at_5", "Success@5"),
    ]
    lines = [
        "| System | P@5 | P@10 | nDCG@10 | MRR | Success@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if metrics_by_system.empty:
        lines.append("| n/a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |")
        return lines
    best_values = {
        metric: float(metrics_by_system[metric].max())
        for metric, _ in columns
        if metric != "system" and metric in metrics_by_system.columns
    }
    for _, row in metrics_by_system.iterrows():
        cells = [str(row["system"])]
        for metric, _ in columns[1:]:
            value = float(row[metric])
            text = format_float(value)
            if math.isclose(value, best_values[metric], rel_tol=1e-12, abs_tol=1e-12):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def best_by_query_type(metrics_by_query_type: pd.DataFrame, metric: str = "ndcg_at_10") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_type, group in metrics_by_query_type.groupby("type", sort=True):
        if group.empty:
            continue
        best_value = float(group[metric].max())
        tied_systems = group[
            group[metric].astype(float).map(lambda value: math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12))
        ]["system"].astype(str)
        rows.append({"type": query_type, "system": ", ".join(tied_systems), metric: best_value})
    return pd.DataFrame(rows)


def build_warning_lines(context: dict[str, Any]) -> list[str]:
    warnings = list(context.get("warnings", []))
    if not warnings:
        return ["- None"]
    return [f"- {warning}" for warning in warnings]


def write_markdown_reports(
    output_dir: Path,
    *,
    metrics_by_query: pd.DataFrame,
    metrics_by_system: pd.DataFrame,
    metrics_by_query_type: pd.DataFrame,
    joined: pd.DataFrame,
    context: dict[str, Any],
) -> None:
    (output_dir / "README.md").write_text(build_readme(), encoding="utf-8")
    (output_dir / "system_comparison.md").write_text(
        build_system_comparison(metrics_by_system, metrics_by_query_type, context),
        encoding="utf-8",
    )
    (output_dir / "failure_analysis.md").write_text(
        build_failure_analysis(metrics_by_query, metrics_by_query_type, joined, context),
        encoding="utf-8",
    )
    (output_dir / "evaluation_summary.md").write_text(
        build_evaluation_summary(metrics_by_system, context),
        encoding="utf-8",
    )
    (output_dir / "presentation_summary.md").write_text(
        build_presentation_summary(metrics_by_system, metrics_by_query_type, context),
        encoding="utf-8",
    )


def build_readme() -> str:
    return r"""# Search Quality Metrics

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
"""


def build_system_comparison(
    metrics_by_system: pd.DataFrame,
    metrics_by_query_type: pd.DataFrame,
    context: dict[str, Any],
) -> str:
    lines = [
        "# System Comparison",
        "",
        "This experiment compares retrieval systems using shared human relevance judgments.",
        "",
        "## Evaluation Setup",
        "",
        f"- Evaluated queries: {context['query_count']}",
        f"- Evaluated systems: {context['system_count']}",
        "- Relevance scale: 2 = highly relevant, 1 = partially relevant, 0 = not relevant",
        f"- Binary relevance threshold: relevance >= {context['relevance_threshold']}",
        f"- Evaluation mode: {'incomplete/preliminary' if context['allow_incomplete_labels'] else 'complete/strict'}",
        "",
        "## Main Comparison",
        "",
    ]
    lines.extend(build_main_comparison_table(metrics_by_system))
    lines.extend(["", "## Best System by Metric", ""])
    for metric in ["precision_at_5", "precision_at_10", "ndcg_at_10", "mrr", "success_at_5"]:
        lines.append(f"- `{metric}`: `{best_system(metrics_by_system, metric)}`")

    lines.extend(["", "## Best System by Query Type", ""])
    best_type_rows = best_by_query_type(metrics_by_query_type)
    if best_type_rows.empty:
        lines.append("- n/a")
    else:
        for _, row in best_type_rows.iterrows():
            lines.append(f"- `{row['type']}`: `{row['system']}` (nDCG@10={format_float(row['ndcg_at_10'])})")

    lines.extend(
        [
            "",
            "## Quality Trade-Offs",
            "",
            "Metric differences should be interpreted with the labeling status and query count in mind.",
            "When systems tie or differ only by very small margins, the result should be treated as inconclusive until labels are complete.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(build_warning_lines(context))
    return "\n".join(lines) + "\n"


def build_failure_analysis(
    metrics_by_query: pd.DataFrame,
    metrics_by_query_type: pd.DataFrame,
    joined: pd.DataFrame,
    context: dict[str, Any],
) -> str:
    lines = ["# Failure Analysis", ""]
    lines.extend(["## Lowest-Performing Queries", ""])
    if metrics_by_query.empty:
        lines.append("- n/a")
    else:
        query_avg = (
            metrics_by_query.groupby(["query_id", "query", "type"], as_index=False)["ndcg_at_10"]
            .mean()
            .sort_values("ndcg_at_10", ascending=True, kind="stable")
            .head(10)
        )
        for _, row in query_avg.iterrows():
            lines.append(
                f"- `{row['query_id']}` ({row['type']}): {row['query']} - average nDCG@10={format_float(row['ndcg_at_10'])}."
            )

    lines.extend(["", "## Largest System Differences", ""])
    if metrics_by_query.empty:
        lines.append("- n/a")
    else:
        gaps = []
        for (query_id, query, query_type), group in metrics_by_query.groupby(["query_id", "query", "type"]):
            if group.empty:
                continue
            best = float(group["ndcg_at_10"].max())
            worst = float(group["ndcg_at_10"].min())
            gaps.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "type": query_type,
                    "gap": best - worst,
                    "best_systems": ",".join(group[group["ndcg_at_10"] == best]["system"].astype(str)),
                    "worst_systems": ",".join(group[group["ndcg_at_10"] == worst]["system"].astype(str)),
                    "best": best,
                    "worst": worst,
                }
            )
        for item in sorted(gaps, key=lambda row: row["gap"], reverse=True)[:10]:
            lines.append(
                f"- `{item['query_id']}` ({item['type']}): gap={format_float(item['gap'])}; "
                f"best `{item['best_systems']}`={format_float(item['best'])}, "
                f"worst `{item['worst_systems']}`={format_float(item['worst'])}. Query: {item['query']}"
            )

    lines.extend(["", "## Reranking Regressions", ""])
    comparisons = [
        ("faiss_baseline", "faiss_style_rerank"),
        ("qdrant_semantic", "qdrant_filtered"),
        ("qdrant_semantic", "qdrant_object_rerank"),
    ]
    regression_count = 0
    for baseline, candidate in comparisons:
        pivot = metrics_by_query.pivot_table(
            index=["query_id", "query", "type"],
            columns="system",
            values="ndcg_at_10",
            aggfunc="first",
        )
        if baseline not in pivot.columns or candidate not in pivot.columns:
            continue
        regressions = pivot[pivot[candidate] < pivot[baseline]].copy()
        regressions["delta"] = regressions[candidate] - regressions[baseline]
        for index, row in regressions.sort_values("delta", ascending=True).head(10).iterrows():
            query_id, query, query_type = index
            regression_count += 1
            lines.append(
                f"- `{query_id}` ({query_type}): `{candidate}` lowered nDCG@10 versus `{baseline}` "
                f"({format_float(row[candidate])} vs {format_float(row[baseline])}, delta={format_float(row['delta'])}). "
                f"Query: {query}"
            )
    if regression_count == 0:
        lines.append("- No reranking/filtering regressions detected from current labels.")

    lines.extend(["", "## Query-Type Weaknesses", ""])
    if metrics_by_query_type.empty:
        lines.append("- n/a")
    else:
        overall = metrics_by_query.groupby("system")["ndcg_at_10"].mean().to_dict()
        weakness_count = 0
        for _, row in metrics_by_query_type.iterrows():
            system_average = float(overall.get(row["system"], 0.0))
            if float(row["ndcg_at_10"]) < system_average:
                weakness_count += 1
                lines.append(
                    f"- `{row['system']}` underperforms its overall nDCG@10 on `{row['type']}` "
                    f"({format_float(row['ndcg_at_10'])} vs overall {format_float(system_average)})."
                )
        if weakness_count == 0:
            lines.append("- No query-type weaknesses detected from current labels.")

    comment_rows = joined[joined["comment"].astype(str).str.strip() != ""]
    lines.extend(["", "## Manual Label Comments Used", ""])
    if comment_rows.empty:
        lines.append("- No manual comments are available in the current labels.")
    else:
        for _, row in comment_rows[["query_id", "system", "rank", "image_id", "comment"]].drop_duplicates().head(10).iterrows():
            lines.append(
                f"- `{row['query_id']}` `{row['system']}` rank {row['rank']} image `{row['image_id']}`: {row['comment']}"
            )

    lines.extend(["", "## Warnings", ""])
    lines.extend(build_warning_lines(context))
    return "\n".join(lines) + "\n"


def build_evaluation_summary(metrics_by_system: pd.DataFrame, context: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Summary",
        "",
        f"- Run timestamp: {context['timestamp']}",
        f"- Results input: `{context['results_path']}`",
        f"- Labels input: `{context['labels_path']}`",
        f"- Number of queries: {context['query_count']}",
        f"- Number of systems: {context['system_count']}",
        f"- Number of result rows: {context['result_row_count']}",
        f"- Number of joined relevance labels: {context['joined_label_count']}",
        f"- Missing-label count: {context['missing_label_count']}",
        f"- Relevance threshold: {context['relevance_threshold']}",
        f"- Evaluation status: {'incomplete' if context['allow_incomplete_labels'] else 'complete'}",
        f"- Best system by nDCG@10: `{best_system(metrics_by_system, 'ndcg_at_10')}`",
        f"- Best system by MRR: `{best_system(metrics_by_system, 'mrr')}`",
        f"- Best system by Precision@10: `{best_system(metrics_by_system, 'precision_at_10')}`",
        "",
        "## Generated Output Files",
        "",
    ]
    for filename in context["generated_files"]:
        lines.append(f"- `{filename}`")
    lines.extend(["", "## Warnings", ""])
    lines.extend(build_warning_lines(context))
    return "\n".join(lines) + "\n"


def build_presentation_summary(
    metrics_by_system: pd.DataFrame,
    metrics_by_query_type: pd.DataFrame,
    context: dict[str, Any],
) -> str:
    lines = [
        "# Search Quality Evaluation",
        "",
    ]
    if context["allow_incomplete_labels"]:
        lines.extend(["Preliminary results based on incomplete relevance judgments.", ""])
    lines.extend(
        [
            f"- Evaluated {context['query_count']} queries across {context['system_count']} retrieval systems.",
            "- Relevance scale: 2 highly relevant, 1 partially relevant, 0 not relevant.",
            f"- Binary metrics use relevance >= {context['relevance_threshold']} as relevant.",
            f"- Best overall system by nDCG@10: `{best_system(metrics_by_system, 'ndcg_at_10')}`.",
            "",
        ]
    )
    lines.extend(build_main_comparison_table(metrics_by_system))
    lines.extend(["", "## Best System by Query Group", ""])
    best_type_rows = best_by_query_type(metrics_by_query_type)
    if best_type_rows.empty:
        lines.append("- n/a")
    else:
        for _, row in best_type_rows.iterrows():
            lines.append(f"- `{row['type']}`: `{row['system']}` (nDCG@10={format_float(row['ndcg_at_10'])})")
    lines.extend(["", "Shared human judgments make the system comparison fair because each query-image pair is labeled once and reused across retrieval methods."])
    return "\n".join(lines) + "\n"


def validate_joined_labels(
    joined: pd.DataFrame,
    *,
    allow_incomplete_labels: bool,
) -> None:
    missing = joined[joined["relevance_text"] == ""]
    if missing.empty or allow_incomplete_labels:
        return
    grouped = (
        missing.groupby(["query_id", "system"])
        .size()
        .reset_index(name="missing_labels")
        .sort_values(["query_id", "system"])
    )
    examples = grouped.head(20).to_dict("records")
    raise ValueError(
        "Missing relevance labels for result rows. Run with --allow-incomplete-labels for a preliminary run, "
        f"or complete labels first. Missing by query/system examples: {examples}"
    )


def build_context(
    *,
    results_path: Path,
    labels_path: Path,
    output_dir: Path,
    results: pd.DataFrame,
    joined: pd.DataFrame,
    data_quality: dict[str, Any],
    rank_quality: dict[str, Any],
    relevance_threshold: int,
    allow_incomplete_labels: bool,
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    missing_label_count = int((joined["relevance_text"] == "").sum())
    warnings: list[str] = []
    if allow_incomplete_labels:
        warnings.append(
            "Evaluation was run with --allow-incomplete-labels; metrics use only labeled result rows and are preliminary."
        )
    if missing_label_count:
        warnings.append(f"{missing_label_count} result rows are missing completed relevance labels.")
    for name, frame in [
        ("result rows without judgments", data_quality["result_rows_without_judgments"]),
        ("judgments not appearing in any result", data_quality["judgments_not_in_results"]),
        ("duplicated system/query/image rows", rank_quality["duplicate_system_query_image_rows"]),
        ("duplicate ranks within a query/system", rank_quality["duplicate_ranks"]),
        ("rank gaps", rank_quality["rank_gaps"]),
    ]:
        if not frame.empty:
            warnings.append(f"Detected {len(frame)} {name}. See evaluation logs and source CSVs for details.")

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "results_path": str(results_path.relative_to(PROJECT_ROOT) if results_path.is_relative_to(PROJECT_ROOT) else results_path),
        "labels_path": str(labels_path.relative_to(PROJECT_ROOT) if labels_path.is_relative_to(PROJECT_ROOT) else labels_path),
        "output_dir": str(output_dir),
        "query_count": int(results["query_id"].nunique()),
        "system_count": int(results["system"].nunique()),
        "result_row_count": int(len(results)),
        "joined_label_count": int((joined["relevance_text"] != "").sum()),
        "missing_label_count": missing_label_count,
        "relevance_threshold": relevance_threshold,
        "allow_incomplete_labels": allow_incomplete_labels,
        "bootstrap_samples": bootstrap_samples,
        "random_seed": random_seed,
        "generated_files": [
            "README.md",
            "metrics_by_system.csv",
            "metrics_by_query.csv",
            "metrics_by_query_type.csv",
            "metrics_statistical_summary.csv",
            "metrics_confidence_intervals.csv",
            "system_comparison.md",
            "failure_analysis.md",
            "presentation_summary.md",
            "evaluation_summary.md",
        ],
        "warnings": warnings,
    }


def run_evaluation(
    *,
    results_path: Path,
    labels_path: Path,
    output_dir: Path,
    relevance_threshold: int = 1,
    allow_incomplete_labels: bool = False,
    bootstrap_samples: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    if relevance_threshold not in {1, 2}:
        raise ValueError("--relevance-threshold must be 1 or 2")

    results_path = resolve_project_path(results_path)
    labels_path = resolve_project_path(labels_path)
    output_dir = resolve_project_path(output_dir)

    results = load_results(results_path)
    judgments = load_judgments(labels_path)
    label_errors = validate_relevance_labels(judgments)
    if label_errors:
        raise ValueError("; ".join(label_errors))

    rank_quality = validate_result_ranks(results)
    joined, data_quality = join_results_with_labels(results, judgments)
    validate_joined_labels(joined, allow_incomplete_labels=allow_incomplete_labels)

    metrics_by_query = evaluate_all_queries(
        joined,
        relevance_threshold=relevance_threshold,
        allow_incomplete_labels=allow_incomplete_labels,
    )
    metrics_by_system = aggregate_by_system(metrics_by_query)
    metrics_by_query_type = aggregate_by_query_type(metrics_by_query)
    statistical_summary = build_statistical_summary(metrics_by_query)
    confidence_intervals = bootstrap_confidence_intervals(
        metrics_by_query,
        samples=bootstrap_samples,
        random_seed=random_seed,
    )

    context = build_context(
        results_path=results_path,
        labels_path=labels_path,
        output_dir=output_dir,
        results=results,
        joined=joined,
        data_quality=data_quality,
        rank_quality=rank_quality,
        relevance_threshold=relevance_threshold,
        allow_incomplete_labels=allow_incomplete_labels,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )

    write_csv_outputs(
        output_dir,
        metrics_by_query,
        metrics_by_system,
        metrics_by_query_type,
        statistical_summary,
        confidence_intervals,
    )
    write_markdown_reports(
        output_dir,
        metrics_by_query=metrics_by_query,
        metrics_by_system=metrics_by_system,
        metrics_by_query_type=metrics_by_query_type,
        joined=joined,
        context=context,
    )

    context["best_ndcg10"] = best_system(metrics_by_system, "ndcg_at_10")
    context["best_mrr"] = best_system(metrics_by_system, "mrr")
    context["best_precision10"] = best_system(metrics_by_system, "precision_at_10")
    return context


def main() -> None:
    args = parse_args()
    context = run_evaluation(
        results_path=args.results_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        relevance_threshold=args.relevance_threshold,
        allow_incomplete_labels=args.allow_incomplete_labels,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    print(f"Evaluated queries: {context['query_count']}")
    print(f"Systems evaluated: {context['system_count']}")
    print(f"Result rows: {context['result_row_count']}")
    print(f"Joined relevance labels: {context['joined_label_count']}")
    print(f"Missing labels: {context['missing_label_count']}")
    print(f"Best system by nDCG@10: {context['best_ndcg10']}")
    print(f"Saved outputs: {resolve_project_path(args.output_dir)}")
    if context["warnings"]:
        print("Warnings:")
        for warning in context["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
