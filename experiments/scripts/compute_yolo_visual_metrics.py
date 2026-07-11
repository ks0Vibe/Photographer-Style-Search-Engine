from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


STAGE_DIR = PROJECT_ROOT / "experiments" / "06_yolo_object_retrieval"
VISUAL_INSPECTION_PATH = STAGE_DIR / "visual_inspection.csv"
RETRIEVAL_METRICS_PATH = STAGE_DIR / "retrieval_metrics.csv"
VISUAL_METRICS_PATH = STAGE_DIR / "visual_metrics.csv"
VISUAL_GROUP_METRICS_PATH = STAGE_DIR / "visual_group_metrics.csv"
AUTOMATIC_VS_VISUAL_PATH = STAGE_DIR / "automatic_vs_visual_metrics.csv"
OBJECT_PRECISION_PATH = STAGE_DIR / "object_precision_metrics.csv"
OBJECT_PRECISION_SUMMARY_PATH = STAGE_DIR / "object_precision_summary.csv"
REPORT_PATH = STAGE_DIR / "report.md"

TOP_K = 10
METRIC_COLUMNS = [
    "query",
    "query_group",
    "mode",
    "result_count",
    "precision_at_10",
    "avg_relevance",
    "dcg_at_10",
    "ndcg_at_10",
    "mrr_at_10",
]
GROUP_COLUMNS = [
    "query_group",
    "mode",
    "query_mode_count",
    "precision_at_10",
    "avg_relevance",
    "dcg_at_10",
    "ndcg_at_10",
    "mrr_at_10",
]
OBJECT_COLUMNS = [
    "query",
    "query_group",
    "requested_object",
    "mode",
    "result_count",
    "labeled_object_results",
    "object_precision_at_10",
]
OBJECT_SUMMARY_COLUMNS = [
    "mode",
    "query_count",
    "labeled_object_results",
    "object_precision_at_10",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_visual_relevance(value: str) -> int | None:
    value = str(value or "").strip()
    if value == "":
        return None
    try:
        parsed = int(float(value))
    except ValueError:
        return None
    if parsed not in {0, 1, 2}:
        return None
    return parsed


def parse_object_present(value: str) -> int | None:
    value = str(value or "").strip().lower()
    if value in {"0", "1"}:
        return int(value)
    return None


def dcg_at_k(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances))


def metrics_for_group(rows: list[dict[str, str]]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: int(row["rank"]))
    relevances = [parse_visual_relevance(row.get("visual_relevance", "")) or 0 for row in ordered[:TOP_K]]
    relevant_count = sum(1 for value in relevances if value > 0)
    dcg = dcg_at_k(relevances)
    ideal_dcg = dcg_at_k(sorted(relevances, reverse=True))
    rr = 0.0
    for index, relevance in enumerate(relevances, start=1):
        if relevance > 0:
            rr = 1.0 / index
            break
    return {
        "result_count": len(relevances),
        "precision_at_10": relevant_count / TOP_K,
        "avg_relevance": statistics.fmean(relevances) if relevances else 0.0,
        "dcg_at_10": dcg,
        "ndcg_at_10": dcg / ideal_dcg if ideal_dcg else 0.0,
        "mrr_at_10": rr,
    }


def complete_top_10(rows: list[dict[str, str]]) -> bool:
    by_rank = {int(row["rank"]): row for row in rows if str(row.get("rank", "")).isdigit()}
    for rank in range(1, TOP_K + 1):
        row = by_rank.get(rank)
        if row is None or parse_visual_relevance(row.get("visual_relevance", "")) is None:
            return False
    return True


def compute_visual_metrics(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["query"], row["query_group"], row["mode"])].append(row)

    metrics_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for (query, query_group, mode), group_rows in sorted(grouped.items()):
        if not complete_top_10(group_rows):
            missing.append(f"{query} / {mode}")
            continue
        metrics_rows.append(
            {
                "query": query,
                "query_group": query_group,
                "mode": mode,
                **metrics_for_group(group_rows),
            }
        )
    return metrics_rows, missing


def compute_object_precision(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[str]]:
    allowed_modes = {"qdrant_semantic", "qdrant_object", "qdrant_object_rerank"}
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("mode") in allowed_modes and row.get("requested_object"):
            grouped[(row["query"], row["query_group"], row["requested_object"], row["mode"])].append(row)

    metrics_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for (query, query_group, requested_object, mode), group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: int(row["rank"]))[:TOP_K]
        labels = [parse_object_present(row.get("object_present", "")) for row in ordered]
        if len(ordered) != TOP_K or any(label is None for label in labels):
            missing.append(f"{query} / {mode}")
            continue
        metrics_rows.append({
            "query": query,
            "query_group": query_group,
            "requested_object": requested_object,
            "mode": mode,
            "result_count": len(ordered),
            "labeled_object_results": sum(labels),
            "object_precision_at_10": sum(labels) / TOP_K,
        })
    return metrics_rows, missing


def aggregate_object_precision(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)
    output = []
    for mode, mode_rows in sorted(grouped.items()):
        output.append({
            "mode": mode,
            "query_count": len(mode_rows),
            "labeled_object_results": sum(int(row["labeled_object_results"]) for row in mode_rows),
            "object_precision_at_10": statistics.fmean(float(row["object_precision_at_10"]) for row in mode_rows),
        })
    return output


def aggregate_group_metrics(metrics_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in metrics_rows:
        grouped[(str(row["query_group"]), str(row["mode"]))].append(row)

    output = []
    for (query_group, mode), rows in sorted(grouped.items()):
        aggregate = {
            "query_group": query_group,
            "mode": mode,
            "query_mode_count": len(rows),
        }
        for metric in ("precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10"):
            aggregate[metric] = statistics.fmean(float(row[metric]) for row in rows)
        output.append(aggregate)
    return output


def build_automatic_vs_visual(visual_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    automatic_rows = read_csv(RETRIEVAL_METRICS_PATH)
    visual_by_key = {
        (str(row["query"]), str(row["mode"])): row
        for row in visual_rows
    }
    output = []
    for automatic in automatic_rows:
        key = (str(automatic.get("query", "")), str(automatic.get("mode", "")))
        row = visual_by_key.get(key)
        auto_precision = automatic.get("precision_at_10", "")
        auto_ndcg = automatic.get("ndcg_at_10", "")
        visual_precision = float(row["precision_at_10"]) if row else ""
        visual_ndcg = float(row["ndcg_at_10"]) if row else ""
        output.append(
            {
                "query": automatic.get("query", ""),
                "query_group": automatic.get("query_group", ""),
                "mode": automatic.get("mode", ""),
                "auto_precision_at_10": auto_precision,
                "visual_precision_at_10": visual_precision,
                "auto_ndcg_at_10": auto_ndcg,
                "visual_ndcg_at_10": visual_ndcg,
                "precision_delta": _delta(auto_precision, visual_precision),
                "ndcg_delta": _delta(auto_ndcg, visual_ndcg),
                "interpretation": "visual labels complete" if row else "visual inspection pending",
            }
        )
    return output


def _delta(raw_auto: object, visual_value: object) -> str:
    if visual_value == "":
        return ""
    try:
        return f"{float(visual_value) - float(raw_auto):.4f}"
    except (TypeError, ValueError):
        return ""


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def update_report(metrics_rows: list[dict[str, object]], group_rows: list[dict[str, object]], missing: list[str], object_rows: list[dict[str, object]], object_missing: list[str]) -> None:
    if not REPORT_PATH.exists():
        return

    report = REPORT_PATH.read_text(encoding="utf-8")
    marker = "\n## Visual Inspection Metrics\n"
    report = report.split(marker)[0].rstrip()

    if metrics_rows:
        section = [
            "## Visual Inspection Metrics",
            "",
            "Human visual labels are available for at least one complete query/mode top-10 group. These metrics use `visual_relevance` from `visual_inspection.csv`.",
            "",
            "Overall visual metrics:",
            "",
            markdown_table(metrics_rows, METRIC_COLUMNS),
            "",
            "Visual metrics by query group:",
            "",
            markdown_table(group_rows, GROUP_COLUMNS),
        ]
    else:
        section = [
            "## Visual Inspection Metrics",
            "",
            "Human visual inspection is pending. `visual_inspection.csv` exists for manual labeling, but no query/mode pair has complete top-10 visual labels yet.",
        ]

    if missing:
        section.extend(["", "Incomplete or missing top-10 visual labels:", ""])
        section.extend(f"- {item}" for item in missing[:80])
        if len(missing) > 80:
            section.append(f"- ... {len(missing) - 80} more")

    section.extend([
        "",
        "## Manual Object Precision@10",
        "",
        "This metric uses the independently inspected `object_present` field. It is not the automatic YOLO payload metric.",
        "",
    ])
    object_summary = aggregate_object_precision(object_rows)
    if object_summary:
        section.append(markdown_table(object_summary, OBJECT_SUMMARY_COLUMNS))
    else:
        section.append("No complete object-present labels are available yet.")
    if object_missing:
        section.extend(["", "Incomplete object labels:", ""])
        section.extend(f"- {item}" for item in object_missing[:80])

    REPORT_PATH.write_text(report + "\n\n" + "\n".join(section) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_csv(VISUAL_INSPECTION_PATH)
    if not rows:
        print(f"No visual inspection rows found: {VISUAL_INSPECTION_PATH}")
        write_csv(VISUAL_METRICS_PATH, [], METRIC_COLUMNS)
        write_csv(VISUAL_GROUP_METRICS_PATH, [], GROUP_COLUMNS)
        write_csv(AUTOMATIC_VS_VISUAL_PATH, [], [
            "query",
            "query_group",
            "mode",
            "auto_precision_at_10",
            "visual_precision_at_10",
            "auto_ndcg_at_10",
            "visual_ndcg_at_10",
            "precision_delta",
            "ndcg_delta",
            "interpretation",
        ])
        write_csv(OBJECT_PRECISION_PATH, [], OBJECT_COLUMNS)
        write_csv(OBJECT_PRECISION_SUMMARY_PATH, [], OBJECT_SUMMARY_COLUMNS)
        return

    metrics_rows, missing = compute_visual_metrics(rows)
    group_rows = aggregate_group_metrics(metrics_rows)
    comparison_rows = build_automatic_vs_visual(metrics_rows)
    object_rows, object_missing = compute_object_precision(rows)
    object_summary = aggregate_object_precision(object_rows)

    write_csv(VISUAL_METRICS_PATH, metrics_rows, METRIC_COLUMNS)
    write_csv(VISUAL_GROUP_METRICS_PATH, group_rows, GROUP_COLUMNS)
    write_csv(
        AUTOMATIC_VS_VISUAL_PATH,
        comparison_rows,
        [
            "query",
            "query_group",
            "mode",
            "auto_precision_at_10",
            "visual_precision_at_10",
            "auto_ndcg_at_10",
            "visual_ndcg_at_10",
            "precision_delta",
            "ndcg_delta",
            "interpretation",
        ],
    )
    write_csv(OBJECT_PRECISION_PATH, object_rows, OBJECT_COLUMNS)
    write_csv(OBJECT_PRECISION_SUMMARY_PATH, object_summary, OBJECT_SUMMARY_COLUMNS)
    update_report(metrics_rows, group_rows, missing, object_rows, object_missing)

    print(f"Complete visual metric groups: {len(metrics_rows)}")
    print(f"Incomplete query/mode pairs: {len(missing)}")
    for item in missing[:20]:
        print(f"- missing labels: {item}")
    if len(missing) > 20:
        print(f"- ... {len(missing) - 20} more")
    print(f"Wrote: {VISUAL_METRICS_PATH}")
    print(f"Wrote: {VISUAL_GROUP_METRICS_PATH}")
    print(f"Wrote: {AUTOMATIC_VS_VISUAL_PATH}")
    print(f"Complete object precision groups: {len(object_rows)}")
    print(f"Incomplete object precision groups: {len(object_missing)}")
    print(f"Wrote: {OBJECT_PRECISION_PATH}")
    print(f"Wrote: {OBJECT_PRECISION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
