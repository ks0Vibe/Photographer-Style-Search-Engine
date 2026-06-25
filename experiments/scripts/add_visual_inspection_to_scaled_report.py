from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = PROJECT_ROOT / "experiments" / "05_scaled_retrieval_quality"

RETRIEVAL_RESULTS_PATH = STAGE_DIR / "retrieval_results.csv"
RETRIEVAL_METRICS_PATH = STAGE_DIR / "retrieval_metrics.csv"
DATASET_STATS_PATH = STAGE_DIR / "dataset_stats.json"
PAYLOAD_STATS_PATH = STAGE_DIR / "qdrant_payload_stats.json"
LATENCY_RESULTS_PATH = STAGE_DIR / "latency_results.csv"

VISUAL_INSPECTION_PATH = STAGE_DIR / "visual_inspection.csv"
VISUAL_METRICS_PATH = STAGE_DIR / "visual_metrics.csv"
VISUAL_GROUP_METRICS_PATH = STAGE_DIR / "visual_group_metrics.csv"
AUTOMATIC_VS_VISUAL_PATH = STAGE_DIR / "automatic_vs_visual_metrics.csv"
QUALITATIVE_FINDINGS_PATH = STAGE_DIR / "qualitative_findings.md"
REPORT_PATH = STAGE_DIR / "report.md"

TOP_K = 10


# Rank-level visual labels from manual inspection of all PNG result grids in
# experiments/05_scaled_retrieval_quality/visualizations/.
VISUAL_RELEVANCE: dict[tuple[str, str], list[int]] = {
    ("warm cinematic landscape", "qdrant_semantic"): [1, 1, 1, 1, 1, 1, 1, 2, 2, 1],
    ("warm cinematic landscape", "qdrant_rerank"): [2, 2, 1, 1, 1, 1, 2, 2, 2, 1],
    ("dark moody forest", "qdrant_semantic"): [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ("dark moody forest", "qdrant_rerank"): [1, 2, 2, 2, 2, 2, 1, 2, 1, 2],
    ("vibrant summer beach", "qdrant_semantic"): [2, 2, 2, 2, 2, 2, 2, 2, 1, 2],
    ("vibrant summer beach", "qdrant_rerank"): [2, 1, 2, 2, 2, 2, 1, 2, 2, 1],
    ("minimal architecture", "qdrant_semantic"): [1, 2, 2, 1, 2, 2, 2, 2, 2, 2],
    ("minimal architecture", "qdrant_rerank"): [2, 1, 2, 2, 2, 1, 1, 1, 2, 2],
    ("person", "qdrant_semantic"): [1, 0, 2, 2, 1, 0, 0, 0, 2, 2],
    ("person", "qdrant_keyword"): [1, 2, 2, 1, 2, 2, 1, 2, 2, 2],
    ("car", "qdrant_semantic"): [2, 2, 2, 1, 0, 1, 2, 1, 2, 1],
    ("car", "qdrant_keyword"): [2, 2, 1, 1, 2, 1, 2, 1, 1, 2],
    ("dog", "qdrant_semantic"): [2, 2, 2, 2, 2, 2, 2, 1, 2, 2],
    ("dog", "qdrant_keyword"): [2, 2, 2, 2, 2, 2, 2, 1, 2, 2],
    ("building", "qdrant_semantic"): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ("building", "qdrant_keyword"): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ("person in street photography", "qdrant_semantic"): [2, 2, 2, 2, 2, 1, 2, 2, 2, 2],
    ("person in street photography", "qdrant_keyword_primary"): [2, 2, 2, 2, 2, 1, 2, 2, 2, 2],
    ("person in street photography", "qdrant_keyword_secondary"): [2, 2, 2, 1, 2, 1, 2, 1, 2, 2],
    ("car at night", "qdrant_semantic"): [2, 2, 2, 1, 2, 1, 1, 1, 1, 2],
    ("car at night", "qdrant_keyword_primary"): [2, 2, 2, 2, 2, 2, 1, 2, 2, 2],
    ("car at night", "qdrant_keyword_secondary"): [2, 2, 1, 1, 1, 2, 2, 1, 1, 2],
    ("dog on beach", "qdrant_semantic"): [2, 2, 2, 1, 2, 1, 2, 1, 1, 2],
    ("dog on beach", "qdrant_keyword_primary"): [2, 2, 2, 1, 2, 2, 1, 1, 2, 2],
    ("dog on beach", "qdrant_keyword_secondary"): [2, 2, 2, 1, 2, 1, 2, 1, 1, 2],
    ("dark forest with fog", "qdrant_semantic"): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ("dark forest with fog", "qdrant_keyword_primary"): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ("dark forest with fog", "qdrant_rerank"): [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
}

GRID_OBSERVATIONS: dict[tuple[str, str], str] = {
    ("warm cinematic landscape", "qdrant_semantic"): "Mostly landscape imagery, but many results are cool or dark rather than warm/cinematic.",
    ("warm cinematic landscape", "qdrant_rerank"): "Reranking adds more warm haze, sunset color, and atmospheric landscape consistency.",
    ("dark moody forest", "qdrant_semantic"): "Strong forest retrieval with some variation in how dark and moody the results are.",
    ("dark moody forest", "qdrant_rerank"): "Reranking keeps the forest theme and generally prioritizes darker atmospheric images.",
    ("vibrant summer beach", "qdrant_semantic"): "Very strong beach and coast retrieval with bright turquoise water and summer visual cues.",
    ("vibrant summer beach", "qdrant_rerank"): "Still visually beach-oriented, though a few results are tropical or coastal without a clear beach.",
    ("minimal architecture", "qdrant_semantic"): "Consistent architecture retrieval, with mixed minimalism.",
    ("minimal architecture", "qdrant_rerank"): "More clean geometric compositions appear, but some dramatic building crops are only partial minimal matches.",
    ("person", "qdrant_semantic"): "Several clear people appear, but the grid also includes non-person or highly ambiguous imagery.",
    ("person", "qdrant_keyword"): "The person keyword improves visible human presence, although some people remain small or ambiguous.",
    ("car", "qdrant_semantic"): "Mostly car imagery, with one clear failure and several partial vehicle/car-detail matches.",
    ("car", "qdrant_keyword"): "Keyword filtering keeps vehicles visible, but centrality varies and some results are partial vehicle-context matches.",
    ("dog", "qdrant_semantic"): "Very strong dog retrieval; one result is visually ambiguous as a real dog versus dog-like object.",
    ("dog", "qdrant_keyword"): "The dog keyword grid is visually almost identical to semantic retrieval and mostly contains clear dogs.",
    ("building", "qdrant_semantic"): "Consistently clear building and architecture results.",
    ("building", "qdrant_keyword"): "Keyword filtering is redundant here because semantic retrieval already returns architecture.",
    ("person in street photography", "qdrant_semantic"): "Strong street-photo imagery with people in urban contexts; one portrait-like result weakens the context.",
    ("person in street photography", "qdrant_keyword_primary"): "The person keyword preserves visible people but does not improve the already strong street-photo context.",
    ("person in street photography", "qdrant_keyword_secondary"): "The street keyword preserves urban scenes but sometimes returns streets where the person is small or absent.",
    ("car at night", "qdrant_semantic"): "Strong night atmosphere, but some results are gas stations, roads, or lights without a clear central car.",
    ("car at night", "qdrant_keyword_primary"): "The car keyword visibly improves full-query matches by keeping cars central in night scenes.",
    ("car at night", "qdrant_keyword_secondary"): "The night keyword preserves night scenes but sometimes loses the car.",
    ("dog on beach", "qdrant_semantic"): "Mostly dogs on sand or beach scenes; some results show only generic sand/ground rather than a clear coast.",
    ("dog on beach", "qdrant_keyword_primary"): "The dog keyword keeps dogs visible, but beach context is uneven in the lower ranks.",
    ("dog on beach", "qdrant_keyword_secondary"): "The beach keyword keeps coastal context, but some dogs are small or only partially beach-related.",
    ("dark forest with fog", "qdrant_semantic"): "Excellent dark, foggy forest retrieval; semantic search already satisfies the query visually.",
    ("dark forest with fog", "qdrant_keyword_primary"): "Forest filtering returns strong dark/fog forest images.",
    ("dark forest with fog", "qdrant_rerank"): "Reranking maintains high style consistency, but semantic retrieval was already strong.",
}

STYLE_QUERIES = {
    "warm cinematic landscape",
    "dark moody forest",
    "vibrant summer beach",
    "minimal architecture",
    "dark forest with fog",
}

OBJECT_QUERY_OBJECTS = {
    "person": "person",
    "car": "car",
    "dog": "dog",
    "building": "building",
    "person in street photography": "person",
    "car at night": "car",
    "dog on beach": "dog",
    "dark forest with fog": "forest",
    "warm cinematic landscape": "landscape",
    "dark moody forest": "forest",
    "vibrant summer beach": "beach/coast",
    "minimal architecture": "architecture",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_json_list(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def dcg_at_k(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances))


def compute_metrics(rows: list[dict[str, Any]], relevance_key: str) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: int(row["rank"]))
    relevances = [int(row[relevance_key]) for row in ordered]
    relevant_count = sum(1 for value in relevances if value > 0)
    dcg = dcg_at_k(relevances)
    ideal_dcg = dcg_at_k(sorted(relevances, reverse=True))
    reciprocal_rank = 0.0
    for index, relevance in enumerate(relevances, start=1):
        if relevance > 0:
            reciprocal_rank = 1.0 / index
            break
    return {
        "result_count": float(len(relevances)),
        "precision_at_10": relevant_count / TOP_K,
        "avg_relevance": statistics.fmean(relevances) if relevances else 0.0,
        "dcg_at_10": dcg,
        "ndcg_at_10": dcg / ideal_dcg if ideal_dcg > 0 else 0.0,
        "mrr_at_10": reciprocal_rank,
    }


def aggregate_metrics(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in group_keys)].append(row)

    metric_names = ("precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10")
    output: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        aggregate = {field: value for field, value in zip(group_keys, key, strict=True)}
        aggregate["query_mode_count"] = len(group_rows)
        for metric_name in metric_names:
            aggregate[metric_name] = statistics.fmean(float(row[metric_name]) for row in group_rows)
        output.append(aggregate)
    return output


def build_visual_note(query: str, mode: str, relevance: int) -> str:
    observation = GRID_OBSERVATIONS[(query, mode)]
    if relevance == 2:
        return f"Full visual match. {observation}"
    if relevance == 1:
        return f"Partial visual match. {observation}"
    return f"Visual failure. {observation}"


def failure_reason(query: str, mode: str, relevance: int) -> str:
    if relevance == 2:
        return "good_match"
    if relevance == 1:
        if query in STYLE_QUERIES:
            return "style_mismatch"
        if "keyword" in mode:
            return "partial_match"
        if query in {"person", "car"}:
            return "object_too_small"
        return "partial_match"
    if "keyword" in mode:
        return "keyword_noise"
    if query in {"person", "car", "dog", "building"}:
        return "wrong_object"
    return "wrong_scene"


def visible_main_object(query: str, relevance: int) -> str:
    if relevance == 0:
        return "none clear"
    return OBJECT_QUERY_OBJECTS.get(query, "query subject")


def style_match(query: str, relevance: int) -> str:
    if query not in STYLE_QUERIES and query not in {"car at night", "dog on beach", "person in street photography"}:
        return "not_applicable"
    if relevance == 2:
        return "yes"
    if relevance == 1:
        return "partial"
    return "no"


def build_visual_inspection_rows(retrieval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in retrieval_rows:
        query = row["query"]
        mode = row["mode"]
        rank = int(row["rank"])
        key = (query, mode)
        if key not in VISUAL_RELEVANCE:
            raise KeyError(f"Missing visual relevance labels for {query} / {mode}")
        relevance_values = VISUAL_RELEVANCE[key]
        if rank < 1 or rank > len(relevance_values):
            raise ValueError(f"Unexpected rank {rank} for {query} / {mode}")
        relevance = relevance_values[rank - 1]
        output.append(
            {
                "query": query,
                "query_group": row["query_group"],
                "mode": mode,
                "rank": rank,
                "image_id": row["image_id"],
                "file_path": row["file_path"],
                "visual_relevance": relevance,
                "visual_notes": build_visual_note(query, mode, relevance),
                "visible_main_object": visible_main_object(query, relevance),
                "style_match": style_match(query, relevance),
                "failure_reason": failure_reason(query, mode, relevance),
                "weak_relevance": row["weak_relevance"],
                "ai_description": row["ai_description"],
                "keywords": row["keywords"],
            }
        )
    return output


def compute_visual_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["query"], row["query_group"], row["mode"])].append(row)

    output: list[dict[str, Any]] = []
    for (query, query_group, mode), group_rows in sorted(grouped.items()):
        output.append(
            {
                "query": query,
                "query_group": query_group,
                "mode": mode,
                **compute_metrics(group_rows, "visual_relevance"),
            }
        )
    return output


def compare_automatic_visual(auto_rows: list[dict[str, Any]], visual_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    auto_by_key = {
        (row["query"], row["query_group"], row["mode"]): row
        for row in auto_rows
    }
    semantic_visual = {
        (row["query"], row["query_group"]): row
        for row in visual_rows
        if row["mode"] == "qdrant_semantic"
    }
    output: list[dict[str, Any]] = []
    for visual in visual_rows:
        key = (visual["query"], visual["query_group"], visual["mode"])
        auto = auto_by_key[key]
        auto_precision = float(auto["precision_at_10"])
        visual_precision = float(visual["precision_at_10"])
        auto_ndcg = float(auto["ndcg_at_10"])
        visual_ndcg = float(visual["ndcg_at_10"])
        precision_delta = visual_precision - auto_precision
        ndcg_delta = visual_ndcg - auto_ndcg

        interpretation = "automatic metric agrees with visual inspection"
        if precision_delta < -0.15 or ndcg_delta < -0.15:
            interpretation = "automatic metric overestimates quality"
        elif precision_delta > 0.15 or ndcg_delta > 0.15:
            interpretation = "automatic metric underestimates quality"
        if "keyword" in visual["mode"] and (precision_delta < -0.05 or ndcg_delta < -0.05):
            interpretation = "keyword metadata likely inflated score"
        semantic = semantic_visual.get((visual["query"], visual["query_group"]))
        if visual["mode"] == "qdrant_rerank" and semantic and visual_ndcg > float(semantic["ndcg_at_10"]) + 0.02:
            interpretation = "style reranking visually improves consistency"

        output.append(
            {
                "query": visual["query"],
                "query_group": visual["query_group"],
                "mode": visual["mode"],
                "auto_precision_at_10": auto_precision,
                "visual_precision_at_10": visual_precision,
                "auto_ndcg_at_10": auto_ndcg,
                "visual_ndcg_at_10": visual_ndcg,
                "precision_delta": precision_delta,
                "ndcg_delta": ndcg_delta,
                "interpretation": interpretation,
            }
        )
    return output


def format_float(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.0000"


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    display_rows = rows[:max_rows] if max_rows is not None else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display_rows:
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def top_visual_metric_rows(rows: list[dict[str, Any]], reverse: bool) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row["ndcg_at_10"]), reverse=reverse)[:6]


def build_qualitative_findings(
    visual_rows: list[dict[str, Any]],
    visual_metrics: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> str:
    keyword_help = [
        row for row in visual_metrics
        if "keyword" in row["mode"] and float(row["precision_at_10"]) >= 0.9 and float(row["avg_relevance"]) >= 1.6
    ]
    keyword_fail = [
        row for row in visual_rows
        if "keyword" in row["mode"] and int(row["visual_relevance"]) < 2
    ][:10]
    rerank_help = [
        row for row in comparison_rows
        if row["interpretation"] == "style reranking visually improves consistency"
    ]
    rerank_no_help = [
        row for row in comparison_rows
        if row["mode"] == "qdrant_rerank" and row["interpretation"] != "style reranking visually improves consistency"
    ]
    mismatches = [
        row for row in comparison_rows
        if "overestimates" in row["interpretation"] or "inflated" in row["interpretation"]
    ]

    fail_display = [
        {
            "query": row["query"],
            "mode": row["mode"],
            "rank": row["rank"],
            "image_id": row["image_id"],
            "visual_relevance": row["visual_relevance"],
            "failure_reason": row["failure_reason"],
            "visual_notes": row["visual_notes"][:120],
        }
        for row in keyword_fail
    ]

    lines = [
        "# Qualitative Visual Findings",
        "",
        "## Summary",
        "",
        "Visual inspection was completed for all 28 PNG result grids. The inspection confirms that the scaled pipeline is operational, but it also shows that weak automatic labels are optimistic for keyword-filtered modes because they evaluate metadata terms that keyword filtering also uses.",
        "",
        "## Cases where keyword filtering helps",
        "",
        "Keyword filtering visibly helps when the keyword corresponds to a broad, visually stable concept. `building` returns consistently architectural results, `dog` returns mostly clear dog images, and the primary `car` keyword improves `car at night` by keeping cars central in night scenes.",
        "",
        markdown_table(keyword_help, ["query", "query_group", "mode", "precision_at_10", "avg_relevance", "ndcg_at_10"], 8),
        "",
        "## Cases where keyword filtering fails",
        "",
        "Keyword filtering fails or becomes partial when the keyword preserves only one part of the query. For `car at night`, the secondary `night` keyword returns night scenes, gas stations, and roads where the car is not always central. For `person in street photography`, the `street` keyword can preserve urban context while reducing the prominence of the person.",
        "",
        markdown_table(fail_display, ["query", "mode", "rank", "image_id", "visual_relevance", "failure_reason", "visual_notes"], 10),
        "",
        "## Cases where style reranking helps",
        "",
        "For `warm cinematic landscape`, reranking visibly improves global appearance by moving warmer, hazier, and more atmospheric landscapes higher in the grid.",
        "",
        markdown_table(rerank_help, ["query", "query_group", "mode", "auto_ndcg_at_10", "visual_ndcg_at_10", "interpretation"]),
        "",
        "## Cases where style reranking does not help",
        "",
        "Reranking does not always improve visual relevance. `dark forest with fog` was already very strong under semantic retrieval, so reranking mostly preserved quality. For beach and minimal architecture, reranking changed style emphasis but did not clearly improve every rank.",
        "",
        markdown_table(rerank_no_help, ["query", "query_group", "mode", "auto_ndcg_at_10", "visual_ndcg_at_10", "interpretation"]),
        "",
        "## Evidence that YOLO is needed",
        "",
        "The grids show that metadata keyword coverage is not the same as object-level correctness. Keyword filters can select payloads containing `person`, `car`, `dog`, or `street`, but they cannot verify whether the object is visually central. The `detected_objects` payload is currently empty, so the system has no independent object evidence. YOLO is needed to populate detected objects in SQLite and Qdrant and to support object-aware retrieval and reranking.",
        "",
        "## Automatic vs visual metric mismatch",
        "",
        "The largest mismatches occur where weak automatic labels reward metadata agreement more than visual correctness. These cases should be treated as diagnostics rather than final quality claims.",
        "",
        markdown_table(mismatches, ["query", "query_group", "mode", "auto_precision_at_10", "visual_precision_at_10", "auto_ndcg_at_10", "visual_ndcg_at_10", "interpretation"], 12),
    ]
    return "\n".join(lines) + "\n"


def build_report(
    dataset_stats: dict[str, Any],
    payload_stats: dict[str, Any],
    auto_metrics: list[dict[str, Any]],
    visual_metrics: list[dict[str, Any]],
    visual_group_metrics: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
) -> str:
    overall_auto = aggregate_metrics(auto_metrics, ("mode",))
    overall_visual = aggregate_metrics(visual_metrics, ("mode",))
    latency_display = aggregate_metrics(
        [
            {
                "mode": row["mode"],
                "precision_at_10": row["avg_latency_ms"],
                "avg_relevance": row["min_latency_ms"],
                "dcg_at_10": row["max_latency_ms"],
                "ndcg_at_10": row["avg_latency_ms"],
                "mrr_at_10": row["avg_latency_ms"],
            }
            for row in latency_rows
        ],
        ("mode",),
    )
    latency_table = [
        {
            "mode": row["mode"],
            "query_mode_count": row["query_mode_count"],
            "avg_latency_ms": row["precision_at_10"],
            "avg_min_latency_ms": row["avg_relevance"],
            "avg_max_latency_ms": row["dcg_at_10"],
        }
        for row in latency_display
    ]
    dataset_table = [
        {"stat": key, "value": json.dumps(value) if isinstance(value, list) else value}
        for key, value in dataset_stats.items()
        if key != "qdrant_local_mode_limitation"
    ]
    payload_summary = [
        {"stat": "keyword_coverage", "value": format_float(payload_stats["keyword_coverage"])},
        {"stat": "object_coverage", "value": format_float(payload_stats["object_coverage"])},
        {"stat": "unique_keywords", "value": payload_stats["unique_keywords"]},
        {"stat": "unique_detected_objects", "value": payload_stats["unique_detected_objects"]},
    ]
    selected_comparison = sorted(
        comparison_rows,
        key=lambda row: abs(float(row["ndcg_delta"])),
        reverse=True,
    )[:10]

    lines = [
        "# Scaled Retrieval Quality Evaluation",
        "",
        "## 1. Goal",
        "",
        "This evaluation tests retrieval behavior after scaling the local corpus to approximately 25k images. The goal is not only to verify that CLIP/Qdrant retrieval remains operational, but also to determine whether the returned images actually look relevant when inspected visually.",
        "",
        "## 2. Current System",
        "",
        "The system uses Unsplash Lite images, SQLite metadata, OpenCLIP `ViT-B-32` image/text embeddings, a FAISS `IndexFlatIP` baseline, and a local persistent Qdrant collection named `photos`. Qdrant payloads include metadata keywords and visual descriptors. Style-aware reranking uses brightness, contrast, saturation, warmth, and color histograms.",
        "",
        "## 3. Dataset and Index Statistics",
        "",
        markdown_table(dataset_table, ["stat", "value"]),
        "",
        "The scaled pipeline is technically successful and operational at 24,916 images. All local metadata rows have embeddings and are indexed in FAISS and Qdrant.",
        "",
        "## 4. Payload Diagnostics",
        "",
        markdown_table(payload_summary, ["stat", "value"]),
        "",
        "Top payload keywords:",
        "",
        markdown_table(payload_stats["top_keywords"], ["keyword", "count"], 25),
        "",
        "Qdrant keyword coverage is 100%, but detected-object coverage is 0%. This means keyword search can narrow candidates by metadata, but it cannot verify whether an object is visually present or central. Metadata keywords are broad and over-inclusive; examples include `dog` on rabbit images, `food` on rabbit/plant/horse images, `portrait` on people/dogs/flowers, and `street` on protest signs, aerial houses, or alleys.",
        "",
        "## 5. Evaluation Methodology",
        "",
        "The evaluation uses style/semantic queries, object-like queries, and combined queries. Retrieval modes include Qdrant semantic search, keyword-filtered search, and style reranking where applicable. The original automatic metrics use weak heuristic relevance labels based on metadata terms, detected-object payloads, and simple descriptor thresholds.",
        "",
        "Keyword-filtered modes achieved high weak-label scores, but these scores should be interpreted carefully. Since the weak evaluator checks metadata terms, and keyword filtering also selects by metadata terms, the evaluation partially rewards the retrieval mode for satisfying its own filter condition. Visual inspection is therefore necessary to determine whether the images actually contain the requested object or scene.",
        "",
        "### Visual Inspection Protocol",
        "",
        "All PNG result grids in `visualizations/` were inspected manually. Each rank was assigned a visual relevance label: `2` for a full visual match, `1` for a partial match, and `0` for a visual failure. The labels are stored in `visual_inspection.csv`. Visual metrics use the same formulas as the weak-label metrics: Precision@10, average graded relevance, DCG@10, nDCG@10, and MRR@10.",
        "",
        "## 6. Automatic Weak-Label Results",
        "",
        "Overall automatic metrics:",
        "",
        markdown_table(overall_auto, ["mode", "query_mode_count", "precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10"]),
        "",
        "Latency by mode:",
        "",
        markdown_table(latency_table, ["mode", "query_mode_count", "avg_latency_ms", "avg_min_latency_ms", "avg_max_latency_ms"]),
        "",
        "These automatic scores are useful for regression testing, but they are optimistic for keyword-filtered modes because metadata terms influence both retrieval and evaluation.",
        "",
        "## 7. Visual Inspection Results",
        "",
        "Overall visual metrics:",
        "",
        markdown_table(overall_visual, ["mode", "query_mode_count", "precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10"]),
        "",
        "Visual metrics by query group and mode:",
        "",
        markdown_table(visual_group_metrics, ["query_group", "mode", "query_mode_count", "precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10"]),
        "",
        "Automatic versus visual metrics:",
        "",
        markdown_table(selected_comparison, ["query", "query_group", "mode", "auto_precision_at_10", "visual_precision_at_10", "auto_ndcg_at_10", "visual_ndcg_at_10", "precision_delta", "ndcg_delta", "interpretation"]),
        "",
        "Representative visual evidence:",
        "",
        "![Warm cinematic landscape semantic results](visualizations/warm_cinematic_landscape__qdrant_semantic.png)",
        "",
        "**Figure 1.** Semantic retrieval for `warm cinematic landscape`. The grid retrieves landscapes, but many are cool, dark, or only partially warm/cinematic.",
        "",
        "![Warm cinematic landscape reranked results](visualizations/warm_cinematic_landscape__qdrant_rerank.png)",
        "",
        "**Figure 2.** Reranked retrieval for `warm cinematic landscape`. The reranker improves style consistency by moving warmer, hazier, and more atmospheric landscapes higher in the grid.",
        "",
        "![Dog keyword results](visualizations/dog__qdrant_keyword.png)",
        "",
        "**Figure 3.** Keyword-filtered retrieval for `dog`. This object keyword performs well in the inspected grid, with mostly clear dog images.",
        "",
        "![Person keyword results](visualizations/person__qdrant_keyword.png)",
        "",
        "**Figure 4.** Keyword-filtered retrieval for `person`. The filter improves human presence, but several ranks contain small, ambiguous, or non-central people, showing where automatic metadata scores overestimate visual quality.",
        "",
        "![Person in street photography keyword-secondary results](visualizations/person_in_street_photography__qdrant_keyword_secondary.png)",
        "",
        "**Figure 5.** Keyword-filtered retrieval for `person in street photography` using the `street` keyword. The filter improves urban context, but does not always guarantee a visible central person.",
        "",
        "![Car at night keyword-primary results](visualizations/car_at_night__qdrant_keyword_primary.png)",
        "",
        "**Figure 6.** Keyword-filtered retrieval for `car at night` using the primary `car` keyword. The filter visibly improves full-query matches by keeping cars central in night scenes.",
        "",
        "![Car at night keyword-secondary results](visualizations/car_at_night__qdrant_keyword_secondary.png)",
        "",
        "**Figure 7.** Keyword-filtered retrieval for `car at night` using the secondary `night` keyword. The filter preserves night scenes, but some results lack a clear central car.",
        "",
        "## 8. Discussion",
        "",
        "The automatic metrics are optimistic because weak labels rely on the same metadata fields used by keyword filtering. Visual inspection shows that keyword coverage does not equal object-level correctness. Keyword filters are useful as coarse narrowing mechanisms, especially for broad visual concepts such as `building`, `dog`, and primary `car` intent, but they are not reliable object recognition.",
        "",
        "Style reranking can improve global appearance. This is clearest for `warm cinematic landscape`, where reranking moves warmer and more atmospheric images higher in the grid. It is less useful when semantic retrieval already satisfies the style query, as in `dark forest with fog`, and it cannot verify object presence.",
        "",
        "YOLO is still necessary because the `detected_objects` payload is empty. An object detector would provide independent object-level evidence, allowing the system to verify whether a person, dog, car, or building is visible and central rather than merely mentioned in metadata.",
        "",
        "## 9. Limitations",
        "",
        "- Visual labels were assigned by manual inspection of generated grids, not by a multi-annotator relevance study.",
        "- Automatic labels remain weak diagnostics and are not ground truth.",
        "- Unsplash metadata keywords are noisy and over-inclusive.",
        "- Qdrant is running in local path mode; collections above 20k points can trigger local-mode scalability warnings.",
        "- No object detector currently populates `detected_objects`.",
        "",
        "## 10. Next Steps",
        "",
        "- Add a YOLO object detection pipeline.",
        "- Store detected objects in SQLite and Qdrant payloads.",
        "- Add an object-aware reranker.",
        "- Replace strict keyword filtering with multi-signal scoring.",
        "- Move Qdrant from local mode to Docker/server mode.",
        "- Repeat this evaluation after YOLO is integrated.",
        "",
        "## 11. Conclusion",
        "",
        "The scaled retrieval pipeline is technically successful and operational at 24,916 images. However, visual inspection reveals that retrieval quality is limited by metadata noise. Keyword filtering is useful as a coarse narrowing mechanism, but it is not reliable object recognition. Style reranking improves global visual consistency for style-sensitive queries, but it does not solve object-specific intent. The next necessary step is to add YOLO object detection, store detected objects in SQLite and Qdrant, and repeat the same evaluation with object-aware retrieval.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    retrieval_rows = read_csv(RETRIEVAL_RESULTS_PATH)
    visual_rows = build_visual_inspection_rows(retrieval_rows)
    visual_metrics = compute_visual_metrics(visual_rows)
    visual_group_metrics = aggregate_metrics(visual_metrics, ("query_group", "mode"))
    auto_metrics = read_csv(RETRIEVAL_METRICS_PATH)
    comparison_rows = compare_automatic_visual(auto_metrics, visual_metrics)
    dataset_stats = load_json(DATASET_STATS_PATH)
    payload_stats = load_json(PAYLOAD_STATS_PATH)
    latency_rows = read_csv(LATENCY_RESULTS_PATH)

    write_csv(
        VISUAL_INSPECTION_PATH,
        visual_rows,
        [
            "query",
            "query_group",
            "mode",
            "rank",
            "image_id",
            "file_path",
            "visual_relevance",
            "visual_notes",
            "visible_main_object",
            "style_match",
            "failure_reason",
            "weak_relevance",
            "ai_description",
            "keywords",
        ],
    )
    write_csv(VISUAL_METRICS_PATH, visual_metrics)
    write_csv(VISUAL_GROUP_METRICS_PATH, visual_group_metrics)
    write_csv(AUTOMATIC_VS_VISUAL_PATH, comparison_rows)

    qualitative = build_qualitative_findings(visual_rows, visual_metrics, comparison_rows)
    QUALITATIVE_FINDINGS_PATH.write_text(qualitative, encoding="utf-8")

    report = build_report(
        dataset_stats=dataset_stats,
        payload_stats=payload_stats,
        auto_metrics=auto_metrics,
        visual_metrics=visual_metrics,
        visual_group_metrics=visual_group_metrics,
        comparison_rows=comparison_rows,
        latency_rows=latency_rows,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Updated report: {REPORT_PATH}")
    print(f"Wrote visual inspection CSV: {VISUAL_INSPECTION_PATH}")
    print(f"Wrote visual metrics CSV: {VISUAL_METRICS_PATH}")
    print(f"Wrote visual group metrics CSV: {VISUAL_GROUP_METRICS_PATH}")
    print(f"Wrote automatic-vs-visual CSV: {AUTOMATIC_VS_VISUAL_PATH}")
    print(f"Updated qualitative findings: {QUALITATIVE_FINDINGS_PATH}")


if __name__ == "__main__":
    main()
