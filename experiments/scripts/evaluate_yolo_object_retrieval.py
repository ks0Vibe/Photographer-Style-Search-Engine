from __future__ import annotations

import csv
import json
import math
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.search import ObjectAwareReranker
from app.search.qdrant_config import create_qdrant_client
from scripts.qdrant_common import COLLECTION_NAME, DATABASE_PATH, create_qdrant_service
from scripts.visualize_text_search import build_result_grid, sanitize_query_for_filename


STAGE_DIR = PROJECT_ROOT / "experiments" / "06_yolo_object_retrieval"
VISUALIZATIONS_DIR = STAGE_DIR / "visualizations"

REPORT_PATH = STAGE_DIR / "report.md"
PAYLOAD_STATS_PATH = STAGE_DIR / "object_payload_stats.json"
RETRIEVAL_RESULTS_PATH = STAGE_DIR / "retrieval_results.csv"
RETRIEVAL_METRICS_PATH = STAGE_DIR / "retrieval_metrics.csv"
QUERY_GROUP_METRICS_PATH = STAGE_DIR / "query_group_metrics.csv"
AUTOMATIC_VS_VISUAL_PATH = STAGE_DIR / "automatic_vs_visual_metrics.csv"
VISUAL_TEMPLATE_PATH = STAGE_DIR / "visual_inspection_template.csv"
QUALITATIVE_FINDINGS_PATH = STAGE_DIR / "qualitative_findings.md"

TOP_K = 10
CANDIDATE_POOL_SIZE = 100

OBJECT_QUERIES = ["person", "car", "dog", "cat", "building", "bird"]
COMBINED_QUERIES = [
    "person in street photography",
    "car at night",
    "dog on beach",
    "cat indoors",
    "bird in nature",
    "building in city",
]
QUERY_OBJECT = {
    "person": "person",
    "car": "car",
    "dog": "dog",
    "cat": "cat",
    "building": "building",
    "bird": "bird",
    "person in street photography": "person",
    "car at night": "car",
    "dog on beach": "dog",
    "cat indoors": "cat",
    "bird in nature": "bird",
    "building in city": "building",
}
CONTEXT_TERMS = {
    "person in street photography": ["street", "road", "urban", "city", "sidewalk", "pedestrian"],
    "car at night": ["night", "dark", "light", "neon", "city"],
    "dog on beach": ["beach", "sea", "ocean", "coast", "shore", "sand"],
    "cat indoors": ["indoor", "inside", "room", "home", "house", "interior"],
    "bird in nature": ["nature", "tree", "forest", "outdoors", "wildlife", "branch"],
    "building in city": ["city", "urban", "street", "downtown", "architecture"],
}
MODES = [
    "qdrant_semantic",
    "qdrant_keyword",
    "qdrant_object",
    "qdrant_keyword_object",
    "qdrant_object_rerank",
]


def prepare_dirs() -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        try:
            decoded = json.loads(values)
        except json.JSONDecodeError:
            decoded = [values]
        values = decoded
    if not isinstance(values, list):
        return []
    return sorted({str(value).strip().lower() for value in values if str(value).strip()})


def collect_payload_stats() -> dict[str, Any]:
    client = create_qdrant_client()
    object_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    total = 0
    with_objects = 0
    samples = []
    offset = None
    try:
        info = client.get_collection(COLLECTION_NAME)
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                total += 1
                payload = point.payload or {}
                objects = normalize_list(payload.get("detected_objects"))
                keywords = normalize_list(payload.get("keywords"))
                if objects:
                    with_objects += 1
                    object_counter.update(objects)
                    if len(samples) < 10:
                        samples.append(
                            {
                                "image_id": payload.get("image_id"),
                                "file_path": payload.get("file_path"),
                                "ai_description": payload.get("ai_description"),
                                "detected_objects": objects,
                                "keywords": keywords[:12],
                            }
                        )
                keyword_counter.update(keywords)
            if offset is None:
                break
    finally:
        client.close()

    return {
        "collection": COLLECTION_NAME,
        "qdrant_points": int(getattr(info, "points_count", total)),
        "total_scanned_points": total,
        "images_with_detected_objects": with_objects,
        "object_coverage": with_objects / total if total else 0.0,
        "unique_detected_objects": len(object_counter),
        "top_detected_objects": [
            {"object": label, "count": count}
            for label, count in object_counter.most_common(50)
        ],
        "top_keywords": [
            {"keyword": label, "count": count}
            for label, count in keyword_counter.most_common(20)
        ],
        "sample_images_with_objects": samples,
    }


def load_sqlite_descriptions() -> dict[str, str]:
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        rows = conn.execute(
            """
            SELECT image_id, ai_description
            FROM images
            """
        ).fetchall()
    finally:
        conn.close()

    return {
        str(image_id): str(ai_description or "")
        for image_id, ai_description in rows
    }


def run_mode(service, query: str, mode: str) -> list[dict[str, Any]]:
    requested_object = QUERY_OBJECT[query]
    if mode == "qdrant_semantic":
        return service.search_by_text(text=query, top_k=TOP_K)
    if mode == "qdrant_keyword":
        return service.search_by_text(text=query, top_k=TOP_K, keyword_filter=requested_object)
    if mode == "qdrant_object":
        return service.search_by_text(text=query, top_k=TOP_K, object_filter=requested_object)
    if mode == "qdrant_keyword_object":
        return service.search_by_text(
            text=query,
            top_k=TOP_K,
            keyword_filter=requested_object,
            object_filter=requested_object,
        )
    if mode == "qdrant_object_rerank":
        candidates = service.search_by_text(text=query, top_k=CANDIDATE_POOL_SIZE)
        return ObjectAwareReranker.object_heavy().rerank(
            candidates,
            requested_object=requested_object,
            requested_keyword=requested_object,
        )[:TOP_K]
    raise ValueError(f"Unknown mode: {mode}")


def text_blob(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("ai_description") or ""),
            " ".join(normalize_list(row.get("keywords"))),
            " ".join(normalize_list(row.get("detected_objects"))),
        ]
    ).lower()


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def weak_relevance(query: str, row: dict[str, Any]) -> int:
    requested_object = QUERY_OBJECT[query]
    objects = set(normalize_list(row.get("detected_objects")))
    text = text_blob(row)
    object_detected = requested_object in objects
    object_metadata = requested_object in text

    if query in OBJECT_QUERIES:
        if object_detected and object_metadata:
            return 2
        if object_detected or object_metadata:
            return 1
        return 0

    context_match = has_any(text, CONTEXT_TERMS.get(query, []))
    if object_detected and context_match:
        return 2
    if object_detected or context_match or object_metadata:
        return 1
    return 0


def enrich_results(
    query: str,
    query_group: str,
    mode: str,
    results: list[dict[str, Any]],
    latency_ms: float,
    descriptions_by_image_id: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for rank, result in enumerate(results, start=1):
        row = {
            "query": query,
            "query_group": query_group,
            "mode": mode,
            "rank": rank,
            "image_id": str(result.get("image_id", "")),
            "score": float(result.get("score", 0.0) or 0.0),
            "semantic_score": float(result.get("semantic_score", result.get("score", 0.0)) or 0.0),
            "object_score": float(result.get("object_score", 0.0) or 0.0),
            "keyword_score": float(result.get("keyword_score", 0.0) or 0.0),
            "file_path": result.get("file_path", ""),
            "keywords": normalize_list(result.get("keywords")),
            "detected_objects": normalize_list(result.get("detected_objects")),
            "ai_description": result.get("ai_description", "") or descriptions_by_image_id.get(str(result.get("image_id", "")), ""),
            "latency_ms": latency_ms,
        }
        row["weak_relevance"] = weak_relevance(query, row)
        rows.append(row)
    return rows


def dcg_at_k(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances))


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["rank"]))
    relevances = [int(row["weak_relevance"]) for row in ordered]
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
        "latency_ms": statistics.fmean(float(row["latency_ms"]) for row in rows) if rows else 0.0,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["query"], row["query_group"], row["mode"])].append(row)
    output = []
    for (query, query_group, mode), group_rows in sorted(grouped.items()):
        output.append({"query": query, "query_group": query_group, "mode": mode, **metrics_for_rows(group_rows)})
    return output


def aggregate_metrics(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    metric_names = ("precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10", "latency_ms")
    output = []
    for key, group_rows in sorted(grouped.items()):
        aggregate = {field: value for field, value in zip(keys, key, strict=True)}
        aggregate["query_mode_count"] = len(group_rows)
        for metric_name in metric_names:
            aggregate[metric_name] = statistics.fmean(float(row[metric_name]) for row in group_rows)
        output.append(aggregate)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            for key in ("keywords", "detected_objects"):
                if key in csv_row:
                    csv_row[key] = json.dumps(csv_row[key], ensure_ascii=False)
            writer.writerow(csv_row)


def write_visualization(query: str, mode: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output_path = VISUALIZATIONS_DIR / f"{sanitize_query_for_filename(query)}__{mode}.png"
    grid = build_result_grid(query=f"{query} / {mode}", results=rows)
    grid.save(output_path)


def write_visual_template(rows: list[dict[str, Any]]) -> None:
    template_rows = [
        {
            "query": row["query"],
            "query_group": row["query_group"],
            "mode": row["mode"],
            "rank": row["rank"],
            "image_id": row["image_id"],
            "file_path": row["file_path"],
            "visual_relevance": "",
            "visual_notes": "",
        }
        for row in rows
    ]
    write_csv(VISUAL_TEMPLATE_PATH, template_rows)


def write_automatic_vs_visual(metrics_rows: list[dict[str, Any]]) -> None:
    rows = [
        {
            "query": row["query"],
            "query_group": row["query_group"],
            "mode": row["mode"],
            "auto_precision_at_10": row["precision_at_10"],
            "visual_precision_at_10": "",
            "auto_ndcg_at_10": row["ndcg_at_10"],
            "visual_ndcg_at_10": "",
            "precision_delta": "",
            "ndcg_delta": "",
            "interpretation": "visual inspection pending",
        }
        for row in metrics_rows
    ]
    write_csv(AUTOMATIC_VS_VISUAL_PATH, rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    rows = rows[:max_rows] if max_rows is not None else rows
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(payload_stats: dict[str, Any], metrics_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]]) -> str:
    overall_rows = aggregate_metrics(metrics_rows, ("mode",))
    best = sorted(metrics_rows, key=lambda row: float(row["ndcg_at_10"]), reverse=True)[:8]
    worst = sorted(metrics_rows, key=lambda row: float(row["ndcg_at_10"]))[:8]
    top_objects = payload_stats["top_detected_objects"][:20]

    figures = [
        ("person", "qdrant_object", "Object filter for `person`."),
        ("car at night", "qdrant_object", "Object filter for `car at night`."),
        ("dog on beach", "qdrant_object_rerank", "Object-aware rerank for `dog on beach`."),
        ("bird in nature", "qdrant_keyword_object", "Keyword + object filter for `bird in nature`."),
    ]
    figure_lines = []
    for index, (query, mode, caption) in enumerate(figures, start=1):
        file_name = f"{sanitize_query_for_filename(query)}__{mode}.png"
        if (VISUALIZATIONS_DIR / file_name).exists():
            figure_lines.extend(
                [
                    f"![{caption}](visualizations/{file_name})",
                    "",
                    f"**Figure {index}.** {caption}",
                    "",
                ]
            )

    return "\n".join(
        [
            "# YOLO Object Retrieval Evaluation",
            "",
            "## 1. Goal",
            "",
            "This experiment tests whether adding YOLO detected objects improves object-specific retrieval after the previous 25k-scale evaluation found `object_coverage = 0.0` and showed that object search relied on noisy metadata keywords.",
            "",
            "## 2. Implementation",
            "",
            "YOLO detections are extracted with `ultralytics` using `yolov8n.pt` by default. Detected object labels are stored as normalized JSON lists in `images.detected_objects`, with `detection_model` and `detection_updated_at` metadata. Qdrant payload upload now reads those SQLite values and stores them in the `detected_objects` payload field.",
            "",
            "Before YOLO, object search relied only on noisy keywords. After YOLO, object filters can use independent visual evidence when detections have been extracted and uploaded.",
            "",
            "## 3. Object Payload Statistics",
            "",
            markdown_table(
                [
                    {"stat": "qdrant_points", "value": payload_stats["qdrant_points"]},
                    {"stat": "images_with_detected_objects", "value": payload_stats["images_with_detected_objects"]},
                    {"stat": "object_coverage", "value": payload_stats["object_coverage"]},
                    {"stat": "unique_detected_objects", "value": payload_stats["unique_detected_objects"]},
                ],
                ["stat", "value"],
            ),
            "",
            "Top detected object classes:",
            "",
            markdown_table(top_objects, ["object", "count"], 20),
            "",
            "## 4. Retrieval Modes",
            "",
            "- `qdrant_semantic`: CLIP/Qdrant text retrieval without payload filters.",
            "- `qdrant_keyword`: keyword payload filter using the requested object label.",
            "- `qdrant_object`: detected-object payload filter.",
            "- `qdrant_keyword_object`: both keyword and detected-object filters.",
            "- `qdrant_object_rerank`: semantic candidate pool reranked by semantic, object, keyword, and optional style scores.",
            "",
            "## 5. Evaluation Methodology",
            "",
            "The query set contains object-like and combined object/context queries. Weak labels use detected objects plus metadata context. These labels are more reliable than pre-YOLO keyword-only labels because `detected_objects` is independent from Unsplash metadata keywords, but they are still not human ground truth. Visual grids are generated and `visual_inspection_template.csv` is provided for manual labels.",
            "",
            "## 6. Results",
            "",
            "Overall metrics:",
            "",
            markdown_table(overall_rows, ["mode", "query_mode_count", "precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10", "latency_ms"]),
            "",
            "Query group metrics:",
            "",
            markdown_table(group_rows, ["query_group", "mode", "query_mode_count", "precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10", "latency_ms"]),
            "",
            "Best examples:",
            "",
            markdown_table(best, ["query", "query_group", "mode", "precision_at_10", "avg_relevance", "ndcg_at_10", "latency_ms"]),
            "",
            "Worst examples:",
            "",
            markdown_table(worst, ["query", "query_group", "mode", "precision_at_10", "avg_relevance", "ndcg_at_10", "latency_ms"]),
            "",
            "## 7. Visual Findings",
            "",
            "Visualization grids were generated, but final visual relevance judgments require manual inspection. Use `visual_inspection_template.csv` to record visual labels.",
            "",
            *figure_lines,
            "## 8. Discussion",
            "",
            "Object filtering is expected to improve object-specific retrieval when YOLO coverage is high enough. Keyword filtering can still be noisy because it depends on metadata. Strict object filters can be too narrow when YOLO misses small, occluded, unusual, or non-COCO objects. Object-aware reranking is less brittle because it can preserve semantic candidates while promoting detected-object matches.",
            "",
            "## 9. Limitations",
            "",
            "- YOLO may miss small or unusual objects.",
            "- COCO classes are limited and not open vocabulary.",
            "- No multi-annotator visual labels are included yet.",
            "- Qdrant local mode is a scalability limitation.",
            "- CPU inference may be slow for the full 24,916-image corpus.",
            "",
            "## 10. Next Steps",
            "",
            "- Tune the confidence threshold.",
            "- Store bounding boxes and confidence scores.",
            "- Add object-aware reranking to the main search service.",
            "- Use Docker/server Qdrant.",
            "- Consider GroundingDINO or OWL-ViT for open-vocabulary objects.",
            "",
            "## 11. Conclusion",
            "",
            "YOLO addresses the specific limitation found in the previous report only for images that have been processed and uploaded into Qdrant. Once full-corpus object extraction is complete, object filters and object-aware reranking can use visual evidence instead of relying only on noisy keyword metadata.",
            "",
        ]
    )


def build_qualitative_findings(payload_stats: dict[str, Any], metrics_rows: list[dict[str, Any]]) -> str:
    object_rows = [row for row in metrics_rows if row["mode"] == "qdrant_object"]
    rerank_rows = [row for row in metrics_rows if row["mode"] == "qdrant_object_rerank"]
    return "\n".join(
        [
            "# YOLO Object Retrieval Qualitative Findings",
            "",
            "Visualization grids were generated for manual review. Final visual findings should be filled after inspecting `visual_inspection_template.csv` and the PNG grids.",
            "",
            "## Payload Coverage",
            "",
            f"- Images with detected objects in Qdrant: {payload_stats['images_with_detected_objects']}",
            f"- Object coverage: {payload_stats['object_coverage']:.4f}",
            "",
            "## Object Filter Diagnostics",
            "",
            markdown_table(object_rows, ["query", "query_group", "mode", "precision_at_10", "avg_relevance", "ndcg_at_10"], 12),
            "",
            "## Object-Aware Rerank Diagnostics",
            "",
            markdown_table(rerank_rows, ["query", "query_group", "mode", "precision_at_10", "avg_relevance", "ndcg_at_10"], 12),
            "",
            "## Notes",
            "",
            "Keyword noise can still appear in `qdrant_keyword` results. Object filters depend on YOLO recall and will miss images where YOLO did not detect the requested class.",
            "",
        ]
    )


def main() -> None:
    prepare_dirs()
    payload_stats = collect_payload_stats()
    PAYLOAD_STATS_PATH.write_text(json.dumps(payload_stats, indent=2), encoding="utf-8")
    descriptions_by_image_id = load_sqlite_descriptions()

    queries = [(query, "object_like") for query in OBJECT_QUERIES] + [
        (query, "combined") for query in COMBINED_QUERIES
    ]
    retrieval_rows: list[dict[str, Any]] = []

    service = create_qdrant_service()
    try:
        for query, query_group in queries:
            for mode in MODES:
                start = time.perf_counter()
                results = run_mode(service, query, mode)
                latency_ms = (time.perf_counter() - start) * 1000.0
                rows = enrich_results(query, query_group, mode, results, latency_ms, descriptions_by_image_id)
                retrieval_rows.extend(rows)
                write_visualization(query, mode, rows)
    finally:
        service.close()

    metrics_rows = compute_metrics(retrieval_rows)
    group_rows = aggregate_metrics(metrics_rows, ("query_group", "mode"))

    write_csv(RETRIEVAL_RESULTS_PATH, retrieval_rows)
    write_csv(RETRIEVAL_METRICS_PATH, metrics_rows)
    write_csv(QUERY_GROUP_METRICS_PATH, group_rows)
    write_automatic_vs_visual(metrics_rows)
    write_visual_template(retrieval_rows)
    QUALITATIVE_FINDINGS_PATH.write_text(build_qualitative_findings(payload_stats, metrics_rows), encoding="utf-8")
    REPORT_PATH.write_text(build_report(payload_stats, metrics_rows, group_rows), encoding="utf-8")

    print(f"Generated report: {REPORT_PATH}")
    print(f"Generated retrieval metrics: {RETRIEVAL_METRICS_PATH}")
    print(f"Generated visual inspection template: {VISUAL_TEMPLATE_PATH}")
    print(f"Generated visualizations: {VISUALIZATIONS_DIR}")
    print(f"Object coverage: {payload_stats['object_coverage']:.4f}")


if __name__ == "__main__":
    main()
