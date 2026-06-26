from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder
from scripts.synthetic_qdrant_common import (
    SERVER_QDRANT_STORAGE_PATH,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_STAGE_DIR,
    create_synthetic_qdrant_store,
    directory_size_mb,
    synthetic_collection_name,
    synthetic_qdrant_mode,
    synthetic_qdrant_path,
    synthetic_storage_label,
)


REAL_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
REAL_IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
GENERATION_STATS_PATH = SYNTHETIC_DATA_DIR / "synthetic_generation_stats.json"
UPLOAD_STATS_PATH = SYNTHETIC_STAGE_DIR / "synthetic_upload_stats.json"
DATASET_STATS_PATH = SYNTHETIC_STAGE_DIR / "synthetic_dataset_stats.json"
BENCHMARK_RESULTS_PATH = SYNTHETIC_STAGE_DIR / "synthetic_benchmark_results.csv"
LATENCY_SUMMARY_PATH = SYNTHETIC_STAGE_DIR / "synthetic_latency_summary.csv"
REPORT_PATH = SYNTHETIC_STAGE_DIR / "report.md"

QUERY_SPECS = [
    {"query": "dog on beach", "keyword_filter": "beach", "object_filter": "dog"},
    {"query": "car at night", "keyword_filter": "car", "object_filter": "car"},
    {"query": "person in street photography", "keyword_filter": "person", "object_filter": "person"},
    {"query": "warm cinematic landscape", "keyword_filter": "landscape", "object_filter": "person"},
    {"query": "dark moody forest", "keyword_filter": "forest", "object_filter": "bird"},
    {"query": "minimal architecture", "keyword_filter": "architecture", "object_filter": "person"},
    {"query": "bird in nature", "keyword_filter": "bird", "object_filter": "bird"},
    {"query": "cat indoors", "keyword_filter": "cat", "object_filter": "cat"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the synthetic 500k Qdrant collection.")
    parser.add_argument("--data-dir", type=Path, default=SYNTHETIC_DATA_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-pool-size", type=int, default=100)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vector-query-fallback",
        action="store_true",
        help="Use random real image embeddings instead of text-encoded query vectors.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError("Query vector has zero norm")
    return vector / norm


def build_query_vectors(use_fallback: bool, seed: int) -> dict[str, np.ndarray]:
    if not use_fallback:
        try:
            encoder = CLIPEncoder()
            return {
                spec["query"]: normalize_vector(encoder.encode_text(spec["query"]))
                for spec in QUERY_SPECS
            }
        except Exception as exc:
            print(f"Text query encoding failed; falling back to real image vectors: {exc}", flush=True)

    real_embeddings = np.load(REAL_EMBEDDINGS_PATH, mmap_mode="r", allow_pickle=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(real_embeddings.shape[0], size=len(QUERY_SPECS), replace=False)
    return {
        spec["query"]: normalize_vector(np.asarray(real_embeddings[index], dtype=np.float32))
        for spec, index in zip(QUERY_SPECS, indices.tolist(), strict=True)
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    SYNTHETIC_STAGE_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_path(args.data_dir)
    synthetic_embeddings = np.load(data_dir / "synthetic_embeddings.npy", mmap_mode="r", allow_pickle=False)
    synthetic_image_ids = np.load(data_dir / "synthetic_image_ids.npy", mmap_mode="r", allow_pickle=False)
    real_image_ids = np.load(REAL_IMAGE_IDS_PATH, mmap_mode="r", allow_pickle=False)
    generation_stats = read_json(data_dir / "synthetic_generation_stats.json")
    upload_stats = read_json(UPLOAD_STATS_PATH)

    if synthetic_embeddings.ndim != 2 or synthetic_embeddings.shape[1] != 512:
        raise ValueError(f"Expected synthetic embeddings shape (N, 512), got {synthetic_embeddings.shape}")
    if synthetic_embeddings.shape[0] != synthetic_image_ids.shape[0]:
        raise ValueError("Synthetic embedding and image ID counts differ")

    query_vectors = build_query_vectors(args.vector_query_fallback, seed=args.seed)
    store = create_synthetic_qdrant_store()
    result_rows: list[dict[str, Any]] = []
    try:
        collection_size = store.count()
        for spec in QUERY_SPECS:
            mode_filters = [
                ("qdrant_synthetic_semantic", None),
                ("qdrant_synthetic_keyword_filter", {"keyword_filter": spec["keyword_filter"]}),
                ("qdrant_synthetic_object_filter", {"object_filter": spec["object_filter"]}),
            ]
            for mode, filters in mode_filters:
                latencies: list[float] = []
                result_counts: list[int] = []
                top_scores: list[float] = []
                for _ in range(args.latency_runs):
                    start = time.perf_counter()
                    points = store.search(
                        query_vector=query_vectors[spec["query"]],
                        top_k=args.candidate_pool_size,
                        filters=filters,
                    )
                    latencies.append((time.perf_counter() - start) * 1000.0)
                    result_counts.append(len(points))
                    if points:
                        top_scores.append(float(points[0].score))

                result_rows.append(
                    {
                        "query": spec["query"],
                        "mode": mode,
                        "keyword_filter": spec["keyword_filter"] if filters and "keyword_filter" in filters else "",
                        "object_filter": spec["object_filter"] if filters and "object_filter" in filters else "",
                        "collection_size": int(collection_size),
                        "embedding_dim": int(synthetic_embeddings.shape[1]),
                        "top_k": int(args.top_k),
                        "candidate_pool_size": int(args.candidate_pool_size),
                        "runs": int(args.latency_runs),
                        "result_count_avg": statistics.fmean(result_counts),
                        "top_score_avg": statistics.fmean(top_scores) if top_scores else 0.0,
                        "search_latency_ms_avg": statistics.fmean(latencies),
                        "search_latency_ms_p50": percentile(latencies, 0.50),
                        "search_latency_ms_p95": percentile(latencies, 0.95),
                        "search_latency_ms_max": max(latencies),
                        "qdrant_mode": synthetic_qdrant_mode(),
                    }
                )
    finally:
        store.close()

    measured_storage_size_mb = directory_size_mb(
        synthetic_qdrant_path()
        if synthetic_qdrant_mode() == "local"
        else SERVER_QDRANT_STORAGE_PATH
    )
    storage_size_mb = measured_storage_size_mb or upload_stats.get("storage_size_mb")
    dataset_stats = {
        "real_visual_corpus": int(real_image_ids.shape[0]),
        "synthetic_vector_corpus": int(synthetic_embeddings.shape[0]),
        "collection_size": int(collection_size),
        "embedding_dim": int(synthetic_embeddings.shape[1]),
        "vector_dtype": str(synthetic_embeddings.dtype),
        "approx_vector_memory_mb": float(synthetic_embeddings.shape[0] * synthetic_embeddings.shape[1] * 4 / (1024 * 1024)),
        "collection_name": synthetic_collection_name(),
        "qdrant_mode": synthetic_qdrant_mode(),
        "storage": synthetic_storage_label(),
        "distance_metric": "Cosine",
        "upload_time_seconds": upload_stats.get("upload_time_seconds"),
        "indexing_time_seconds_if_available": upload_stats.get("indexing_time_seconds"),
        "storage_size_mb": storage_size_mb,
        "ram_note_if_available": "RAM was not measured by this script.",
        "generation": generation_stats,
        "important_distinction": (
            "Real visual corpus: 24,916 downloaded images. Synthetic scale corpus: "
            "500,000 generated vector objects. Purpose of synthetic corpus: scalability "
            "and indexing benchmark, not visual relevance evaluation."
        ),
    }

    summary_rows: list[dict[str, Any]] = []
    modes = sorted({row["mode"] for row in result_rows})
    for mode in modes:
        group = [row for row in result_rows if row["mode"] == mode]
        summary_rows.append(
            {
                "mode": mode,
                "query_count": len(group),
                "collection_size": int(collection_size),
                "embedding_dim": int(synthetic_embeddings.shape[1]),
                "top_k": int(args.top_k),
                "candidate_pool_size": int(args.candidate_pool_size),
                "qdrant_mode": synthetic_qdrant_mode(),
                "storage_size_mb": storage_size_mb,
                "upload_time_seconds": upload_stats.get("upload_time_seconds"),
                "indexing_time_seconds_if_available": upload_stats.get("indexing_time_seconds"),
                "search_latency_ms_avg": statistics.fmean(float(row["search_latency_ms_avg"]) for row in group),
                "search_latency_ms_p50": statistics.fmean(float(row["search_latency_ms_p50"]) for row in group),
                "search_latency_ms_p95": statistics.fmean(float(row["search_latency_ms_p95"]) for row in group),
                "search_latency_ms_max": max(float(row["search_latency_ms_max"]) for row in group),
                "ram_note_if_available": "RAM was not measured by this script.",
            }
        )

    return dataset_stats, result_rows, summary_rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
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


def build_report(dataset_stats: dict[str, Any], summary_rows: list[dict[str, Any]]) -> str:
    stats_rows = [
        {"stat": "real visual corpus", "value": "24,916 downloaded images"},
        {"stat": "synthetic vector corpus", "value": f"{dataset_stats['synthetic_vector_corpus']:,} generated vector objects"},
        {"stat": "total benchmark scale", "value": f"{dataset_stats['collection_size']:,} synthetic objects"},
        {"stat": "embedding dim", "value": dataset_stats["embedding_dim"]},
        {"stat": "vector dtype", "value": dataset_stats["vector_dtype"]},
        {"stat": "approx vector memory MB", "value": f"{dataset_stats['approx_vector_memory_mb']:.2f}"},
    ]
    qdrant_rows = [
        {"stat": "collection name", "value": dataset_stats["collection_name"]},
        {"stat": "Qdrant mode", "value": dataset_stats["qdrant_mode"]},
        {"stat": "distance metric", "value": dataset_stats["distance_metric"]},
        {"stat": "vector size", "value": dataset_stats["embedding_dim"]},
        {"stat": "upload time seconds", "value": dataset_stats.get("upload_time_seconds", "n/a")},
        {"stat": "storage size MB", "value": dataset_stats.get("storage_size_mb", "n/a")},
    ]

    return "\n".join(
        [
            "# Synthetic 500k Vector Scale Evaluation",
            "",
            "## 1. Goal",
            "",
            "This experiment addresses the 500k-object scale requirement by adding synthetic vector objects derived from real CLIP embeddings.",
            "",
            "Real visual corpus: 24,916 downloaded images. Synthetic scale corpus: 500,000 generated vector objects. Purpose of synthetic corpus: scalability and indexing benchmark, not visual relevance evaluation.",
            "",
            "## 2. Why Synthetic Objects",
            "",
            "Generating and downloading 500,000 real images is not feasible under the local hardware, storage, and time constraints of this project. Synthetic vectors are therefore used only to test vector database indexing, storage, filtering, and search latency at the required object scale.",
            "",
            "## 3. Generation Method",
            "",
            "- Real normalized CLIP embeddings are used as anchors.",
            "- Each synthetic vector is generated with Gaussian perturbation.",
            "- A configurable subset also mixes in a neighboring real embedding.",
            "- Every generated vector is L2-normalized and stored as `float32`.",
            "- Payload templates are copied from source images, including keywords and detected objects when available.",
            "- Every payload includes `is_synthetic = true` and `synthetic_generation = clip_embedding_perturbation_v1`.",
            "- `file_path` is only a proxy reference to the source real image for debugging.",
            "",
            "## 4. Dataset Statistics",
            "",
            markdown_table(stats_rows, ["stat", "value"]),
            "",
            "## 5. Qdrant Collection",
            "",
            markdown_table(qdrant_rows, ["stat", "value"]),
            "",
            "Docker/server Qdrant is preferred for the 500k synthetic benchmark. Local path mode is supported for reproducibility, but it is less suitable for large collections.",
            "",
            "## 6. Search Benchmark",
            "",
            markdown_table(
                summary_rows,
                [
                    "mode",
                    "query_count",
                    "collection_size",
                    "candidate_pool_size",
                    "qdrant_mode",
                    "search_latency_ms_avg",
                    "search_latency_ms_p50",
                    "search_latency_ms_p95",
                    "search_latency_ms_max",
                ],
            ),
            "",
            "Detailed per-query latency rows are saved in `synthetic_benchmark_results.csv`. Aggregated mode summaries are saved in `synthetic_latency_summary.csv`.",
            "",
            "## 7. Interpretation",
            "",
            "The synthetic objects validate that the vector database can hold and search the required 500k-object scale. Keyword and object filters are checked for operational behavior against copied payload fields. Real-image retrieval quality evaluation remains based on the 24,916 downloaded images, and synthetic results should not be used for visual relevance conclusions.",
            "",
            "## 8. Limitations",
            "",
            "- Synthetic vectors are not independent real photographs.",
            "- Metadata payloads are copied from source images.",
            "- Visual relevance cannot be judged from synthetic duplicates or perturbations.",
            "- A real 500k image dataset would be stronger but requires much more storage, download time, and compute.",
            "",
            "## 9. Conclusion",
            "",
            "The project now includes a real visual retrieval corpus for quality experiments and a synthetic 500k vector corpus for scalability, indexing, latency, and hardware requirement evaluation.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.candidate_pool_size <= 0 or args.latency_runs <= 0:
        raise ValueError("--top-k, --candidate-pool-size, and --latency-runs must be greater than 0")
    if args.candidate_pool_size < args.top_k:
        raise ValueError("--candidate-pool-size must be greater than or equal to --top-k")

    dataset_stats, result_rows, summary_rows = benchmark(args)
    DATASET_STATS_PATH.write_text(json.dumps(dataset_stats, indent=2), encoding="utf-8")
    write_csv(BENCHMARK_RESULTS_PATH, result_rows)
    write_csv(LATENCY_SUMMARY_PATH, summary_rows)
    REPORT_PATH.write_text(build_report(dataset_stats, summary_rows), encoding="utf-8")

    print(f"Generated dataset stats: {DATASET_STATS_PATH}")
    print(f"Generated benchmark rows: {BENCHMARK_RESULTS_PATH}")
    print(f"Generated latency summary: {LATENCY_SUMMARY_PATH}")
    print(f"Generated report: {REPORT_PATH}")
    print()
    print("Synthetic 500k benchmark summary:")
    print(f"- Collection: {dataset_stats['collection_name']}")
    print(f"- Collection size: {dataset_stats['collection_size']}")
    print(f"- Qdrant mode: {dataset_stats['qdrant_mode']}")
    for row in summary_rows:
        print(
            f"- {row['mode']}: avg={float(row['search_latency_ms_avg']):.2f} ms, "
            f"p95={float(row['search_latency_ms_p95']):.2f} ms"
        )


if __name__ == "__main__":
    main()
