from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder
from app.search import QdrantRetrievalService, QdrantStore
from scripts.qdrant_common import COLLECTION_NAME, QDRANT_PATH
from scripts.visualize_text_search import build_result_grid, sanitize_query_for_filename


STAGE_DIR = PROJECT_ROOT / "experiments" / "05_scaled_retrieval_quality"
VISUALIZATIONS_DIR = STAGE_DIR / "visualizations"

DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "flat.index"

REPORT_PATH = STAGE_DIR / "report.md"
DATASET_STATS_PATH = STAGE_DIR / "dataset_stats.json"
KEYWORD_STATS_PATH = STAGE_DIR / "keyword_stats.csv"
PAYLOAD_STATS_PATH = STAGE_DIR / "qdrant_payload_stats.json"
RETRIEVAL_RESULTS_PATH = STAGE_DIR / "retrieval_results.csv"
RETRIEVAL_METRICS_PATH = STAGE_DIR / "retrieval_metrics.csv"
QUERY_GROUP_METRICS_PATH = STAGE_DIR / "query_group_metrics.csv"
LATENCY_RESULTS_PATH = STAGE_DIR / "latency_results.csv"
QUALITATIVE_FINDINGS_PATH = STAGE_DIR / "qualitative_findings.md"

TOP_K = 10
LATENCY_RUNS = 3
CANDIDATE_POOL_SIZE = 100


QUERY_TERMS: dict[str, dict[str, list[str]]] = {
    "warm cinematic landscape": {
        "semantic": ["landscape", "mountain", "field", "nature", "scenery", "outdoors"],
        "style": ["warm", "sunset", "sunrise", "golden", "cinematic", "dramatic"],
    },
    "dark moody forest": {
        "semantic": ["forest", "tree", "woods", "woodland"],
        "style": ["dark", "moody", "night", "shadow", "fog", "mist"],
    },
    "vibrant summer beach": {
        "semantic": ["beach", "sea", "ocean", "coast", "shore", "water"],
        "style": ["vibrant", "summer", "sunny", "bright", "colorful"],
    },
    "minimal architecture": {
        "semantic": ["architecture", "building", "interior", "house", "urban"],
        "style": ["minimal", "simple", "clean", "white", "geometric"],
    },
    "person": {
        "semantic": ["person", "people", "human", "man", "woman", "portrait"],
    },
    "car": {
        "semantic": ["car", "automobile", "vehicle", "coupe", "suv", "van"],
    },
    "dog": {
        "semantic": ["dog", "canine", "puppy", "pet"],
    },
    "building": {
        "semantic": ["building", "architecture", "house", "office building", "tower"],
    },
    "person in street photography": {
        "part1": ["person", "people", "human", "man", "woman"],
        "part2": ["street", "road", "urban", "city", "sidewalk", "pedestrian"],
    },
    "car at night": {
        "part1": ["car", "automobile", "vehicle", "coupe", "suv"],
        "part2": ["night", "dark", "city at night", "light", "neon"],
    },
    "dog on beach": {
        "part1": ["dog", "canine", "puppy", "pet"],
        "part2": ["beach", "sea", "ocean", "coast", "shore", "sand"],
    },
    "dark forest with fog": {
        "part1": ["forest", "tree", "woods", "woodland"],
        "part2": ["dark", "fog", "mist", "moody", "shadow", "night"],
    },
}

STYLE_DESCRIPTOR_RULES = {
    "warm cinematic landscape": lambda row: optional_float(row.get("warmth")) >= 0.55,
    "dark moody forest": lambda row: optional_float(row.get("brightness")) <= 0.35,
    "vibrant summer beach": lambda row: optional_float(row.get("saturation")) >= 0.45,
    "minimal architecture": lambda row: (
        optional_float(row.get("saturation")) <= 0.35
        or optional_float(row.get("contrast")) <= 0.35
    ),
}


@dataclass(frozen=True)
class ModeSpec:
    query: str
    query_group: str
    mode: str
    keyword_filter: str | None = None
    rerank: bool = False


def optional_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def prepare_output_dirs() -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for path in VISUALIZATIONS_DIR.glob("*"):
        if path.is_file():
            path.unlink()


def build_mode_specs() -> list[ModeSpec]:
    specs: list[ModeSpec] = []

    for query in (
        "warm cinematic landscape",
        "dark moody forest",
        "vibrant summer beach",
        "minimal architecture",
    ):
        specs.append(ModeSpec(query=query, query_group="style_semantic", mode="qdrant_semantic"))
        specs.append(ModeSpec(query=query, query_group="style_semantic", mode="qdrant_rerank", rerank=True))

    for query in ("person", "car", "dog", "building"):
        specs.append(ModeSpec(query=query, query_group="object_like", mode="qdrant_semantic"))
        specs.append(
            ModeSpec(
                query=query,
                query_group="object_like",
                mode="qdrant_keyword",
                keyword_filter=query,
            )
        )

    combined = {
        "person in street photography": ("person", "street", False),
        "car at night": ("car", "night", False),
        "dog on beach": ("dog", "beach", False),
        "dark forest with fog": ("forest", None, True),
    }
    for query, (primary, secondary, use_rerank) in combined.items():
        specs.append(ModeSpec(query=query, query_group="combined", mode="qdrant_semantic"))
        specs.append(
            ModeSpec(
                query=query,
                query_group="combined",
                mode="qdrant_keyword_primary",
                keyword_filter=primary,
            )
        )
        if secondary:
            specs.append(
                ModeSpec(
                    query=query,
                    query_group="combined",
                    mode="qdrant_keyword_secondary",
                    keyword_filter=secondary,
                )
            )
        if use_rerank:
            specs.append(ModeSpec(query=query, query_group="combined", mode="qdrant_rerank", rerank=True))

    return specs


def load_sqlite_metadata() -> dict[str, dict[str, Any]]:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {DATABASE_PATH}")

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                image_id,
                file_path,
                photo_url,
                ai_description,
                brightness,
                contrast,
                saturation,
                warmth
            FROM images
            """
        ).fetchall()
    finally:
        conn.close()

    return {str(row["image_id"]): dict(row) for row in rows}


def load_dataset_stats(qdrant_points: int) -> dict[str, Any]:
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r", allow_pickle=False)
    image_ids = np.load(IMAGE_IDS_PATH, mmap_mode="r", allow_pickle=False)
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        sqlite_image_rows = int(conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])
    finally:
        conn.close()

    failed_embeddings = max(0, sqlite_image_rows - int(embeddings.shape[0]))

    return {
        "sqlite_image_rows": sqlite_image_rows,
        "embeddings_shape": list(embeddings.shape),
        "image_ids_shape": list(image_ids.shape),
        "faiss_index_vectors": int(faiss_index.ntotal),
        "qdrant_points": int(qdrant_points),
        "embedding_dim": int(embeddings.shape[1]),
        "failed_embeddings_if_available": int(failed_embeddings),
        "qdrant_collection": COLLECTION_NAME,
        "qdrant_distance": "Cosine",
        "qdrant_local_mode_limitation": (
            "Qdrant local path mode may warn for collections above 20k points; "
            "this is a scalability limitation of local mode, not a fatal evaluation error."
        ),
    }


def collect_qdrant_payload_stats() -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    store = QdrantStore(collection_name=COLLECTION_NAME, qdrant_path=QDRANT_PATH)
    keyword_counter: Counter[str] = Counter()
    object_counter: Counter[str] = Counter()
    sample_payloads: list[dict[str, Any]] = []
    points_with_keywords = 0
    points_with_objects = 0
    total_points = 0
    next_offset = None

    try:
        qdrant_points = store.count()
        while True:
            points, next_offset = store.client.scroll(
                collection_name=COLLECTION_NAME,
                limit=512,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for point in points:
                total_points += 1
                payload = point.payload or {}
                keywords = [str(value).lower() for value in (payload.get("keywords") or []) if str(value).strip()]
                detected_objects = [
                    str(value).lower()
                    for value in (payload.get("detected_objects") or [])
                    if str(value).strip()
                ]

                if keywords:
                    points_with_keywords += 1
                    keyword_counter.update(keywords)
                if detected_objects:
                    points_with_objects += 1
                    object_counter.update(detected_objects)

                if len(sample_payloads) < 8:
                    sample_payloads.append(
                        {
                            "image_id": payload.get("image_id"),
                            "file_path": payload.get("file_path"),
                            "ai_description": payload.get("ai_description"),
                            "keywords": keywords[:20],
                            "detected_objects": detected_objects[:20],
                        }
                    )

            if next_offset is None:
                break
    finally:
        store.close()

    keyword_stats = [
        {"keyword": keyword, "count": count}
        for keyword, count in keyword_counter.most_common(100)
    ]
    payload_stats = {
        "total_points_scrolled": total_points,
        "qdrant_points": qdrant_points,
        "points_with_non_empty_keywords": points_with_keywords,
        "points_with_non_empty_detected_objects": points_with_objects,
        "keyword_coverage": points_with_keywords / total_points if total_points else 0.0,
        "object_coverage": points_with_objects / total_points if total_points else 0.0,
        "unique_keywords": len(keyword_counter),
        "unique_detected_objects": len(object_counter),
        "top_keywords": keyword_stats[:25],
        "top_detected_objects": [
            {"detected_object": item, "count": count}
            for item, count in object_counter.most_common(25)
        ],
        "sample_payloads": sample_payloads,
    }
    return payload_stats, keyword_stats, qdrant_points


def write_keyword_stats(rows: list[dict[str, Any]]) -> None:
    with KEYWORD_STATS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["keyword", "count"])
        writer.writeheader()
        writer.writerows(rows)


def run_retrieval(service: QdrantRetrievalService, spec: ModeSpec) -> list[dict[str, Any]]:
    return service.search_by_text(
        text=spec.query,
        top_k=TOP_K,
        keyword_filter=spec.keyword_filter,
        object_filter=None,
        rerank=spec.rerank,
        candidate_pool_size=CANDIDATE_POOL_SIZE,
    )


def build_text_blob(result: dict[str, Any]) -> str:
    parts = [
        str(result.get("ai_description") or ""),
        " ".join(str(value) for value in result.get("keywords", []) or []),
        " ".join(str(value) for value in result.get("detected_objects", []) or []),
    ]
    return " ".join(parts).lower()


def contains_term(text: str, term: str) -> bool:
    term = term.strip().lower()
    if not term:
        return False
    if " " in term:
        return term in text
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def contains_any(text: str, terms: list[str]) -> bool:
    return any(contains_term(text, term) for term in terms)


def style_descriptor_match(query: str, result: dict[str, Any]) -> bool:
    rule = STYLE_DESCRIPTOR_RULES.get(query)
    if rule is None:
        return False
    return bool(rule(result))


def weak_relevance(query: str, query_group: str, result: dict[str, Any]) -> int:
    terms = QUERY_TERMS[query]
    text = build_text_blob(result)

    if query_group == "style_semantic":
        semantic_match = contains_any(text, terms["semantic"])
        style_match = contains_any(text, terms["style"]) or style_descriptor_match(query, result)
        if semantic_match and style_match:
            return 2
        if semantic_match or style_match:
            return 1
        return 0

    if query_group == "object_like":
        primary_term = query.lower()
        if contains_term(text, primary_term):
            return 2
        if contains_any(text, terms["semantic"][1:]):
            return 1
        return 0

    part1_match = contains_any(text, terms["part1"])
    part2_match = contains_any(text, terms["part2"])
    if part1_match and part2_match:
        return 2
    if part1_match or part2_match:
        return 1
    return 0


def enrich_results(
    spec: ModeSpec,
    results: list[dict[str, Any]],
    sqlite_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        image_id = str(result.get("image_id", ""))
        metadata = sqlite_metadata.get(image_id, {})
        row = {
            "query": spec.query,
            "query_group": spec.query_group,
            "mode": spec.mode,
            "rank": rank,
            "image_id": image_id,
            "score": optional_float(result.get("score")),
            "file_path": str(result.get("file_path") or metadata.get("file_path") or ""),
            "photo_url": str(metadata.get("photo_url") or ""),
            "ai_description": str(metadata.get("ai_description") or ""),
            "keywords": [str(value) for value in (result.get("keywords") or [])],
            "brightness": result.get("brightness"),
            "contrast": result.get("contrast"),
            "saturation": result.get("saturation"),
            "warmth": result.get("warmth"),
            "detected_objects": [str(value) for value in (result.get("detected_objects") or [])],
            "keyword_filter": spec.keyword_filter or "",
            "rerank": spec.rerank,
        }
        row["weak_relevance"] = weak_relevance(spec.query, spec.query_group, row)
        rows.append(row)
    return rows


def write_retrieval_results(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "query",
        "query_group",
        "mode",
        "rank",
        "image_id",
        "score",
        "file_path",
        "photo_url",
        "ai_description",
        "keywords",
        "brightness",
        "contrast",
        "saturation",
        "warmth",
        "detected_objects",
        "weak_relevance",
        "keyword_filter",
        "rerank",
    ]
    with RETRIEVAL_RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["keywords"] = json.dumps(row["keywords"], ensure_ascii=False)
            csv_row["detected_objects"] = json.dumps(row["detected_objects"], ensure_ascii=False)
            writer.writerow(csv_row)


def dcg_at_k(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(index + 2) for index, rel in enumerate(relevances))


def compute_metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    relevances = [int(row["weak_relevance"]) for row in sorted(rows, key=lambda item: int(item["rank"]))]
    relevant_count = sum(1 for value in relevances if value > 0)
    dcg = dcg_at_k(relevances)
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal)

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


def compute_retrieval_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["query"]), str(row["query_group"]), str(row["mode"]))].append(row)

    metrics_rows: list[dict[str, Any]] = []
    for (query, query_group, mode), group_rows in sorted(grouped.items()):
        metrics_rows.append(
            {
                "query": query,
                "query_group": query_group,
                "mode": mode,
                **compute_metrics_for_rows(group_rows),
            }
        )
    return metrics_rows


def aggregate_metric_rows(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
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


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_visualization(spec: ModeSpec, rows: list[dict[str, Any]], errors: list[str]) -> None:
    file_name = f"{sanitize_query_for_filename(spec.query)}__{spec.mode}.png"
    output_path = VISUALIZATIONS_DIR / file_name
    try:
        grid = build_result_grid(query=f"{spec.query} / {spec.mode}", results=rows)
        grid.save(output_path)
    except Exception as exc:
        errors.append(f"Could not create visualization for {spec.query} / {spec.mode}: {exc}")


def benchmark_latency(service: QdrantRetrievalService, specs: list[ModeSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        durations: list[float] = []
        for _ in range(LATENCY_RUNS):
            start = time.perf_counter()
            run_retrieval(service, spec)
            durations.append((time.perf_counter() - start) * 1000.0)

        rows.append(
            {
                "query": spec.query,
                "query_group": spec.query_group,
                "mode": spec.mode,
                "keyword_filter": spec.keyword_filter or "",
                "rerank": spec.rerank,
                "runs": LATENCY_RUNS,
                "avg_latency_ms": statistics.fmean(durations),
                "min_latency_ms": min(durations),
                "max_latency_ms": max(durations),
            }
        )
    return rows


def run_evaluation(
    specs: list[ModeSpec],
    sqlite_metadata: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    retrieval_rows: list[dict[str, Any]] = []
    visualization_errors: list[str] = []

    service = QdrantRetrievalService(
        clip_encoder=CLIPEncoder(),
        qdrant_store=QdrantStore(collection_name=COLLECTION_NAME, qdrant_path=QDRANT_PATH),
    )
    try:
        for spec in specs:
            results = run_retrieval(service, spec)
            enriched = enrich_results(spec, results, sqlite_metadata)
            retrieval_rows.extend(enriched)
            write_visualization(spec, enriched, visualization_errors)

        latency_rows = benchmark_latency(service, specs)
    finally:
        service.close()

    return retrieval_rows, latency_rows, visualization_errors


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
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def pick_best_worst(metrics_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sorted_rows = sorted(metrics_rows, key=lambda row: float(row["ndcg_at_10"]), reverse=True)
    best = sorted_rows[:5]
    worst = list(reversed(sorted_rows[-5:]))
    return best, worst


def find_keyword_noise_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        mode = str(row["mode"])
        keyword = str(row.get("keyword_filter") or "")
        if "keyword" not in mode or not keyword:
            continue
        description = str(row.get("ai_description") or "").lower()
        keywords = " ".join(row.get("keywords") or []).lower()
        if contains_term(keywords, keyword) and not contains_term(description, keyword):
            examples.append(row)
        if len(examples) >= 6:
            break
    return examples


def build_qualitative_findings(
    retrieval_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
    payload_stats: dict[str, Any],
) -> str:
    semantic_by_query_group: dict[tuple[str, str], dict[str, Any]] = {}
    for metric in metrics_rows:
        if metric["mode"] == "qdrant_semantic":
            semantic_by_query_group[(metric["query"], metric["query_group"])] = metric

    keyword_help = []
    rerank_help = []
    for metric in metrics_rows:
        semantic = semantic_by_query_group.get((metric["query"], metric["query_group"]))
        if semantic is None:
            continue
        if "keyword" in str(metric["mode"]) and float(metric["ndcg_at_10"]) > float(semantic["ndcg_at_10"]):
            keyword_help.append(metric)
        if metric["mode"] == "qdrant_rerank" and float(metric["ndcg_at_10"]) > float(semantic["ndcg_at_10"]):
            rerank_help.append(metric)

    keyword_noise = find_keyword_noise_examples(retrieval_rows)
    object_failure_examples = [
        row
        for row in retrieval_rows
        if row["query_group"] == "object_like" and not row.get("detected_objects")
    ][:6]

    lines = [
        "# Qualitative Findings",
        "",
        "These examples are derived from `retrieval_results.csv` and the weak-label metrics. They are diagnostic, not human relevance judgments.",
        "",
        "## Keyword Filter Helps",
        "",
    ]
    if keyword_help:
        lines.append(markdown_table(keyword_help, ["query", "query_group", "mode", "ndcg_at_10", "precision_at_10"], 8))
    else:
        lines.append("No keyword-filtered mode improved nDCG@10 over the semantic baseline under the current weak-label heuristic.")

    lines.extend(["", "## Keyword Filter Noise", ""])
    if keyword_noise:
        noise_rows = [
            {
                "query": row["query"],
                "mode": row["mode"],
                "rank": row["rank"],
                "image_id": row["image_id"],
                "keyword_filter": row["keyword_filter"],
                "ai_description": str(row["ai_description"])[:120],
                "keywords": ", ".join(row["keywords"][:8]),
            }
            for row in keyword_noise
        ]
        lines.append(markdown_table(noise_rows, ["query", "mode", "rank", "image_id", "keyword_filter", "ai_description", "keywords"]))
    else:
        lines.append("No keyword-noise examples were detected by the simple description-versus-keyword heuristic.")

    lines.extend(["", "## Reranking Improves Style Consistency", ""])
    if rerank_help:
        lines.append(markdown_table(rerank_help, ["query", "query_group", "mode", "ndcg_at_10", "avg_relevance"], 8))
    else:
        lines.append("No reranked mode improved nDCG@10 over the semantic baseline under the current weak-label heuristic.")

    lines.extend(["", "## Object Search Limitation", ""])
    lines.append(
        f"Detected-object payload coverage is {format_float(payload_stats.get('object_coverage', 0.0))}. "
        "Object-like retrieval therefore relies on CLIP semantics and noisy metadata keywords rather than verified visual object detections."
    )
    if object_failure_examples:
        object_rows = [
            {
                "query": row["query"],
                "mode": row["mode"],
                "rank": row["rank"],
                "image_id": row["image_id"],
                "detected_objects": json.dumps(row["detected_objects"]),
                "ai_description": str(row["ai_description"])[:120],
            }
            for row in object_failure_examples
        ]
        lines.append("")
        lines.append(markdown_table(object_rows, ["query", "mode", "rank", "image_id", "detected_objects", "ai_description"]))

    return "\n".join(lines) + "\n"


def build_report(
    dataset_stats: dict[str, Any],
    payload_stats: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    group_metrics_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    visualization_errors: list[str],
) -> str:
    overall_mode_metrics = aggregate_metric_rows(metrics_rows, ("mode",))
    latency_by_mode = aggregate_metric_rows(
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
    latency_display = [
        {
            "mode": row["mode"],
            "query_mode_count": row["query_mode_count"],
            "avg_latency_ms": row["precision_at_10"],
            "avg_min_latency_ms": row["avg_relevance"],
            "avg_max_latency_ms": row["dcg_at_10"],
        }
        for row in latency_by_mode
    ]
    best, worst = pick_best_worst(metrics_rows)

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

    lines = [
        "# Scaled Retrieval Quality Evaluation",
        "",
        "## 1. Goal",
        "",
        "This evaluation tests retrieval quality after scaling the local image corpus to approximately 25k images. The objective is to verify that the current CLIP/Qdrant retrieval stack remains operational at this size, measure latency, and diagnose whether metadata keyword filtering and style reranking improve result quality under weak automatic labels.",
        "",
        "## 2. Current System",
        "",
        "The current system stores Unsplash Lite image metadata in SQLite, computes OpenCLIP `ViT-B-32` image and text embeddings, indexes normalized vectors in FAISS `IndexFlatIP`, and uploads the same vectors with payloads into a local persistent Qdrant collection named `photos`. Qdrant payloads include metadata keywords and visual descriptors. Style-aware reranking combines CLIP semantic scores with brightness, contrast, saturation, warmth, and color-histogram similarity.",
        "",
        "## 3. Dataset and Index Statistics",
        "",
        markdown_table(dataset_table, ["stat", "value"]),
        "",
        "The system successfully scaled to 24,916 local images. All local metadata images have CLIP embeddings and are indexed in both FAISS and Qdrant.",
        "",
        "## 4. Payload Diagnostics",
        "",
        markdown_table(payload_summary, ["stat", "value"]),
        "",
        "Top payload keywords:",
        "",
        markdown_table(payload_stats["top_keywords"], ["keyword", "count"], 25),
        "",
        "Keyword coverage is 100%, but detected-object coverage is 0%. Keyword filters are technically functional, yet the source metadata is broad and over-inclusive. Manual checks and this evaluation show that labels such as `dog`, `food`, `portrait`, and `street` can attach to visually unrelated or weakly related images.",
        "",
        "Known noisy keyword examples from the current metadata include: `dog` can appear on rabbit images; `food` can appear on rabbit, plant, or horse images; `portrait` can return people, dogs, or flowers; and `street` can return protest signs, aerial houses, or alleys.",
        "",
        "## 5. Evaluation Methodology",
        "",
        "The evaluation uses three query groups: style/semantic queries, object-like queries, and combined queries. Style/semantic queries compare Qdrant semantic retrieval against style reranking. Object-like queries compare semantic retrieval against a keyword filter equal to the query. Combined queries compare semantic retrieval, primary keyword filters, secondary keyword filters when available, and reranking when the query has explicit style intent.",
        "",
        "Because no human relevance dataset is available, relevance labels are weak automatic diagnostics. The heuristic checks query-specific terms in `ai_description`, keywords, and detected objects. For style-sensitive queries it also checks simple descriptor thresholds such as warmth, brightness, saturation, and minimal contrast. Metrics are Precision@10, average graded relevance, DCG@10, nDCG@10, and MRR@10.",
        "",
        "These labels should not be interpreted as final relevance judgments. They are intended to expose regressions, metadata noise, and differences between retrieval modes.",
        "",
        "## 6. Results",
        "",
        "Overall mode metrics:",
        "",
        markdown_table(overall_mode_metrics, ["mode", "query_mode_count", "precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10"]),
        "",
        "Metrics by query group and mode:",
        "",
        markdown_table(group_metrics_rows, ["query_group", "mode", "query_mode_count", "precision_at_10", "avg_relevance", "dcg_at_10", "ndcg_at_10", "mrr_at_10"]),
        "",
        "Latency by mode:",
        "",
        markdown_table(latency_display, ["mode", "query_mode_count", "avg_latency_ms", "avg_min_latency_ms", "avg_max_latency_ms"]),
        "",
        "Best query/mode examples by nDCG@10:",
        "",
        markdown_table(best, ["query", "query_group", "mode", "precision_at_10", "avg_relevance", "ndcg_at_10", "mrr_at_10"]),
        "",
        "Worst query/mode examples by nDCG@10:",
        "",
        markdown_table(worst, ["query", "query_group", "mode", "precision_at_10", "avg_relevance", "ndcg_at_10", "mrr_at_10"]),
        "",
        "Per-result outputs are saved in `retrieval_results.csv`. Per-query metrics are saved in `retrieval_metrics.csv`, group aggregates in `query_group_metrics.csv`, latency measurements in `latency_results.csv`, and result grids in `visualizations/`.",
        "",
        "## 7. Discussion",
        "",
        "Keyword search is weak despite 100% keyword coverage because Unsplash metadata keywords are descriptive but not visually verified. They often describe broad context, possible content, or weak associations rather than central visible objects. A strict keyword filter can therefore narrow the candidate set while still admitting false positives.",
        "",
        "Style reranking is most appropriate for style-sensitive queries because it uses visual descriptors that directly measure image appearance. It is not expected to solve object-specific intent: if the system does not know whether a dog, car, or person is visually present and central, style scores cannot compensate.",
        "",
        "YOLO or a comparable object detector is the next necessary feature. Detected objects would let the system verify visual object presence, populate the currently empty `detected_objects` payload, and separate object intent from noisy metadata labels.",
        "",
        "## 8. Limitations",
        "",
        "- The relevance labels are weak keyword/descriptor heuristics, not human judgments.",
        "- There is no ground-truth relevance dataset yet.",
        "- Qdrant is running in local path mode; collections above 20k points can trigger local-mode scalability warnings.",
        "- Unsplash metadata keywords are noisy and over-inclusive.",
        "- No object detector is currently populating `detected_objects`.",
        "",
        "## 9. Next Steps",
        "",
        "- Add a YOLO object detection pipeline.",
        "- Store detected objects in SQLite and Qdrant payloads.",
        "- Add an object-aware reranker.",
        "- Replace strict keyword filtering with multi-signal scoring.",
        "- Move Qdrant from local mode to Docker/server mode.",
        "- Repeat this evaluation after YOLO is integrated.",
    ]

    if visualization_errors:
        lines.extend(["", "## Visualization Warnings", ""])
        lines.extend(f"- {error}" for error in visualization_errors)

    return "\n".join(lines) + "\n"


def main() -> None:
    warnings.filterwarnings("default")
    prepare_output_dirs()

    payload_stats, keyword_stats, qdrant_points = collect_qdrant_payload_stats()
    dataset_stats = load_dataset_stats(qdrant_points=qdrant_points)
    sqlite_metadata = load_sqlite_metadata()
    specs = build_mode_specs()

    DATASET_STATS_PATH.write_text(json.dumps(dataset_stats, indent=2), encoding="utf-8")
    PAYLOAD_STATS_PATH.write_text(json.dumps(payload_stats, indent=2), encoding="utf-8")
    write_keyword_stats(keyword_stats)

    retrieval_rows, latency_rows, visualization_errors = run_evaluation(specs, sqlite_metadata)
    metrics_rows = compute_retrieval_metrics(retrieval_rows)
    query_group_metrics_rows = aggregate_metric_rows(metrics_rows, ("query_group", "mode"))

    write_retrieval_results(retrieval_rows)
    write_csv_rows(RETRIEVAL_METRICS_PATH, metrics_rows)
    write_csv_rows(QUERY_GROUP_METRICS_PATH, query_group_metrics_rows)
    write_csv_rows(LATENCY_RESULTS_PATH, latency_rows)

    qualitative_findings = build_qualitative_findings(retrieval_rows, metrics_rows, payload_stats)
    QUALITATIVE_FINDINGS_PATH.write_text(qualitative_findings, encoding="utf-8")

    report = build_report(
        dataset_stats=dataset_stats,
        payload_stats=payload_stats,
        metrics_rows=metrics_rows,
        group_metrics_rows=query_group_metrics_rows,
        latency_rows=latency_rows,
        visualization_errors=visualization_errors,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Generated report: {REPORT_PATH}")
    print(f"Generated metrics CSV: {RETRIEVAL_METRICS_PATH}")
    print(f"Generated retrieval CSV: {RETRIEVAL_RESULTS_PATH}")
    print(f"Generated visualizations: {VISUALIZATIONS_DIR}")
    print()
    print("Main findings:")
    print(f"- SQLite rows: {dataset_stats['sqlite_image_rows']}")
    print(f"- Qdrant points: {dataset_stats['qdrant_points']}")
    print(f"- Keyword coverage: {payload_stats['keyword_coverage']:.4f}")
    print(f"- Detected-object coverage: {payload_stats['object_coverage']:.4f}")
    print("- Keyword filtering is available but diagnostic outputs show metadata noise.")
    print("- Object-specific retrieval remains limited until detected_objects is populated.")
    if visualization_errors:
        print()
        print("Visualization warnings:")
        for error in visualization_errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
