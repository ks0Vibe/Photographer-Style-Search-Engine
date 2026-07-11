from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder


EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
QUERIES_PATH = PROJECT_ROOT / "experiments" / "08_validation_set" / "queries.csv"
LABELS_PATH = PROJECT_ROOT / "experiments" / "10_relevance_labeling" / "relevance_judgments.csv"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "12_vector_optimization"
ARTIFACT_DIR = PROJECT_ROOT / "data" / "optimized_vectors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vector optimization strategies for CLIP retrieval."
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--pca-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--write-artifacts", action="store_true")
    parser.add_argument("--docker-container", default="photographer-style-qdrant")
    parser.add_argument("--skip-docker-stats", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def load_text_queries() -> list[dict[str, str]]:
    rows = read_csv_rows(QUERIES_PATH)
    return [
        row
        for row in rows
        if row.get("type") != "image_to_image"
        and not row.get("query", "").startswith("REPLACE_WITH_REAL_IMAGE_ID")
    ]


def encode_queries(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    encoder = CLIPEncoder()
    return {
        row["query_id"]: normalize_vector(encoder.encode_text(row["query"]))
        for row in rows
    }


def top_k_from_scores(scores: np.ndarray, image_ids: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
    if top_k >= scores.shape[0]:
        order = np.argsort(-scores)
    else:
        partition = np.argpartition(-scores, top_k - 1)[:top_k]
        order = partition[np.argsort(-scores[partition])]
    ids = [str(image_ids[index]) for index in order[:top_k]]
    values = [float(scores[index]) for index in order[:top_k]]
    return ids, values


def search_dense(matrix: np.ndarray, query: np.ndarray, image_ids: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
    query_vector = np.asarray(query, dtype=matrix.dtype)
    scores = matrix @ query_vector
    return top_k_from_scores(np.asarray(scores, dtype=np.float32), image_ids, top_k)


def search_int8(
    quantized: np.ndarray,
    scales: np.ndarray,
    query: np.ndarray,
    image_ids: np.ndarray,
    top_k: int,
    chunk_size: int = 8192,
) -> tuple[list[str], list[float]]:
    scores = np.empty(quantized.shape[0], dtype=np.float32)
    query_vector = np.asarray(query, dtype=np.float32)
    for start in range(0, quantized.shape[0], chunk_size):
        end = min(start + chunk_size, quantized.shape[0])
        chunk = quantized[start:end].astype(np.float32) * scales[start:end, None]
        scores[start:end] = chunk @ query_vector
    return top_k_from_scores(scores, image_ids, top_k)


def fit_pca(embeddings: np.ndarray, max_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mean = embeddings.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = embeddings.astype(np.float32, copy=False) - mean
    covariance = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:max_dim]
    return mean, eigenvectors[:, order].astype(np.float32)


def project_pca(vectors: np.ndarray, mean: np.ndarray, components: np.ndarray, dim: int) -> np.ndarray:
    projected = (vectors.astype(np.float32, copy=False) - mean) @ components[:, :dim]
    return normalize_rows(projected)


def quantize_int8_per_vector(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(matrix), axis=1).astype(np.float32)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(matrix / scales[:, None]), -127, 127).astype(np.int8)
    return quantized, scales


def load_relevance_labels() -> dict[tuple[str, str], int]:
    labels: dict[tuple[str, str], int] = {}
    for row in read_csv_rows(LABELS_PATH):
        raw = row.get("relevance", "").strip()
        if raw in {"0", "1", "2"}:
            labels[(row.get("query_id", ""), row.get("image_id", ""))] = int(raw)
    return labels


def precision_at_k(relevances: list[int], threshold: int = 1) -> float:
    if not relevances:
        return 0.0
    return sum(1 for value in relevances if value >= threshold) / len(relevances)


def ndcg_at_k(relevances: list[int]) -> float:
    if not relevances:
        return 0.0
    dcg = sum((2.0**rel - 1.0) / math.log2(index + 2) for index, rel in enumerate(relevances))
    ideal = sum((2.0**rel - 1.0) / math.log2(index + 2) for index, rel in enumerate(sorted(relevances, reverse=True)))
    return dcg / ideal if ideal else 0.0


def timed_search(search_fn, latency_runs: int) -> tuple[list[str], list[float], float]:
    ids: list[str] = []
    scores: list[float] = []
    durations: list[float] = []
    for _ in range(latency_runs):
        start = time.perf_counter()
        ids, scores = search_fn()
        durations.append((time.perf_counter() - start) * 1000.0)
    return ids, scores, float(sum(durations) / len(durations))


def capture_docker_stats(container_name: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{json .}}",
                container_name,
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}

    line = result.stdout.strip().splitlines()
    if not line:
        return {"available": False, "error": "docker stats returned no rows"}
    try:
        parsed = json.loads(line[0])
    except json.JSONDecodeError:
        parsed = {"raw": line[0]}
    parsed["available"] = True
    return parsed


def bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024 * 1024)


def build_report(
    summary_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    docker_stats: dict[str, Any],
) -> str:
    columns = [
        "variant",
        "dim",
        "dtype",
        "vector_memory_25k_mb",
        "estimated_vector_memory_500k_mb",
        "memory_reduction_vs_fp32",
        "avg_latency_ms",
        "avg_overlap_at_10",
        "labeled_precision_at_10",
        "labeled_ndcg_at_10",
    ]
    primary_rows = [row for row in summary_rows if not str(row.get("variant", "")).startswith("pca")]
    pca_rows = [row for row in summary_rows if str(row.get("variant", "")).startswith("pca")]
    lines = [
        "# Vector Optimization Evaluation",
        "",
        "## Goal",
        "",
        "This experiment evaluates vector optimization options for the CLIP retrieval stack. It compares the original 512-dimensional `float32` vectors against lower-memory representations and measures memory footprint, search latency, and ranking similarity to the baseline.",
        "",
        "## Summary",
        "",
        "### Primary production candidates",
        "",
        markdown_table(primary_rows, columns),
        "",
        "### Secondary PCA diagnostic",
        "",
        "PCA variants are shown separately because the observed overlap@10 is very low; they are not a production recommendation until that quality loss is explained and independently validated.",
        "",
        markdown_table(pca_rows, columns),
        "",
        "## Docker Measurement",
        "",
    ]
    if docker_stats.get("available"):
        lines.extend(
            [
                f"- Container: `{docker_stats.get('Name', docker_stats.get('Container', 'n/a'))}`",
                f"- CPU: `{docker_stats.get('CPUPerc', 'n/a')}`",
                f"- Memory: `{docker_stats.get('MemUsage', 'n/a')}`",
                f"- Memory percent: `{docker_stats.get('MemPerc', 'n/a')}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- Docker stats were not available during this run.",
                f"- Error: `{docker_stats.get('error', 'n/a')}`",
                "",
            ]
        )
    lines.extend(
        [
            "To capture container-level memory during defense preparation, run:",
            "",
            "```powershell",
            "docker compose up -d qdrant",
            "docker stats --no-stream photographer-style-qdrant",
            "```",
            "",
            "## Interpretation",
            "",
            "- `float16_512` keeps the same embedding dimension and should preserve ranking almost exactly while halving vector storage.",
            "- `int8_per_vector_512` gives about 4x vector-memory reduction, but pure Python dequantization can increase latency; in production this should be delegated to Qdrant scalar quantization.",
            "- PCA variants are a secondary diagnostic only: the measured overlap@10 is low and must be explained before they can be treated as a valid production option.",
            "- The 500k memory estimate scales only vector payload memory. Full Docker/Qdrant memory also includes HNSW graph, payload indexes, metadata, WAL, allocator overhead, and container runtime overhead.",
            "",
            "## Outputs",
            "",
            "- `vector_optimization_summary.csv`: aggregated variant comparison.",
            "- `vector_optimization_query_results.csv`: per-query latency, overlap, and labeled metrics.",
            "- `artifact_sizes.json`: file-level sizes for generated optimized vector artifacts.",
            "- `docker_stats.json`: container memory snapshot when Docker is available.",
            "",
            f"Per-query rows generated: {len(query_rows)}",
            "",
        ]
    )
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    output = [
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
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def save_artifact(path: Path, array: np.ndarray, enabled: bool, sizes: dict[str, Any]) -> None:
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    sizes[path.name] = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size_mb": bytes_to_mb(path.stat().st_size),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.latency_runs <= 0:
        raise ValueError("--top-k and --latency-runs must be positive")
    if not args.pca_dims:
        raise ValueError("--pca-dims must include at least one dimension")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False).astype(np.float32, copy=False)
    image_ids = np.load(IMAGE_IDS_PATH, allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    embeddings = normalize_rows(embeddings)

    query_specs = load_text_queries()
    query_vectors = encode_queries(query_specs)
    labels = load_relevance_labels()

    baseline_results: dict[str, list[str]] = {}
    artifact_sizes: dict[str, Any] = {}
    query_rows: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []

    fp32 = embeddings
    fp16 = embeddings.astype(np.float16)
    int8_vectors, int8_scales = quantize_int8_per_vector(embeddings)

    variants.append(
        {
            "variant": "fp32_512_baseline",
            "dim": embeddings.shape[1],
            "dtype": "float32",
            "memory_bytes": embeddings.nbytes,
            "search": lambda query: search_dense(fp32, query, image_ids, args.top_k),
            "project_query": lambda query: query,
        }
    )
    variants.append(
        {
            "variant": "float16_512",
            "dim": embeddings.shape[1],
            "dtype": "float16",
            "memory_bytes": fp16.nbytes,
            "search": lambda query: search_dense(fp16, query, image_ids, args.top_k),
            "project_query": lambda query: query,
        }
    )
    variants.append(
        {
            "variant": "int8_per_vector_512",
            "dim": embeddings.shape[1],
            "dtype": "int8+scale",
            "memory_bytes": int8_vectors.nbytes + int8_scales.nbytes,
            "search": lambda query: search_int8(int8_vectors, int8_scales, query, image_ids, args.top_k),
            "project_query": lambda query: query,
        }
    )

    max_pca_dim = max(args.pca_dims)
    pca_mean, pca_components = fit_pca(embeddings, max_pca_dim)
    save_artifact(ARTIFACT_DIR / "clip_embeddings_fp16.npy", fp16, args.write_artifacts, artifact_sizes)
    save_artifact(ARTIFACT_DIR / "clip_embeddings_int8.npy", int8_vectors, args.write_artifacts, artifact_sizes)
    save_artifact(ARTIFACT_DIR / "clip_embeddings_int8_scales.npy", int8_scales, args.write_artifacts, artifact_sizes)
    save_artifact(ARTIFACT_DIR / "pca_mean.npy", pca_mean, args.write_artifacts, artifact_sizes)
    save_artifact(ARTIFACT_DIR / "pca_components.npy", pca_components, args.write_artifacts, artifact_sizes)

    for dim in sorted(set(args.pca_dims), reverse=True):
        projected = project_pca(embeddings, pca_mean, pca_components, dim)
        projected_fp16 = projected.astype(np.float16)
        save_artifact(ARTIFACT_DIR / f"clip_embeddings_pca{dim}_fp32.npy", projected, args.write_artifacts, artifact_sizes)
        save_artifact(ARTIFACT_DIR / f"clip_embeddings_pca{dim}_fp16.npy", projected_fp16, args.write_artifacts, artifact_sizes)

        variants.append(
            {
                "variant": f"pca{dim}_fp32",
                "dim": dim,
                "dtype": "float32",
                "memory_bytes": projected.nbytes,
                "search": lambda query, matrix=projected: search_dense(matrix, query, image_ids, args.top_k),
                "project_query": lambda query, d=dim: normalize_vector((query.astype(np.float32) - pca_mean) @ pca_components[:, :d]),
            }
        )
        variants.append(
            {
                "variant": f"pca{dim}_fp16",
                "dim": dim,
                "dtype": "float16",
                "memory_bytes": projected_fp16.nbytes,
                "search": lambda query, matrix=projected_fp16: search_dense(matrix, query, image_ids, args.top_k),
                "project_query": lambda query, d=dim: normalize_vector((query.astype(np.float32) - pca_mean) @ pca_components[:, :d]),
            }
        )

    for variant in variants:
        for spec in query_specs:
            query_id = spec["query_id"]
            projected_query = variant["project_query"](query_vectors[query_id])
            ids, scores, latency_ms = timed_search(
                lambda v=variant, q=projected_query: v["search"](q),
                args.latency_runs,
            )
            if variant["variant"] == "fp32_512_baseline":
                baseline_results[query_id] = ids
            baseline_ids = baseline_results.get(query_id, ids)
            overlap = len(set(ids).intersection(baseline_ids)) / args.top_k
            relevances = [labels[(query_id, image_id)] for image_id in ids if (query_id, image_id) in labels]
            query_rows.append(
                {
                    "variant": variant["variant"],
                    "query_id": query_id,
                    "query": spec["query"],
                    "type": spec["type"],
                    "top_k": args.top_k,
                    "latency_ms": round(latency_ms, 4),
                    "overlap_at_10": round(overlap, 4),
                    "labeled_results_at_10": len(relevances),
                    "labeled_precision_at_10": round(precision_at_k(relevances), 4),
                    "labeled_ndcg_at_10": round(ndcg_at_k(relevances), 4),
                    "top1_image_id": ids[0] if ids else "",
                    "top1_score": round(scores[0], 6) if scores else 0.0,
                }
            )

    baseline_memory = next(row["memory_bytes"] for row in variants if row["variant"] == "fp32_512_baseline")
    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        rows = [row for row in query_rows if row["variant"] == variant["variant"]]
        labeled_rows = [row for row in rows if int(row["labeled_results_at_10"]) > 0]
        memory_25k = bytes_to_mb(variant["memory_bytes"])
        estimated_500k = variant["memory_bytes"] / embeddings.shape[0] * 500_000 / (1024 * 1024)
        summary_rows.append(
            {
                "variant": variant["variant"],
                "dim": variant["dim"],
                "dtype": variant["dtype"],
                "vector_memory_25k_mb": round(memory_25k, 4),
                "estimated_vector_memory_500k_mb": round(estimated_500k, 4),
                "memory_reduction_vs_fp32": round(baseline_memory / variant["memory_bytes"], 4),
                "avg_latency_ms": round(sum(float(row["latency_ms"]) for row in rows) / len(rows), 4),
                "avg_overlap_at_10": round(sum(float(row["overlap_at_10"]) for row in rows) / len(rows), 4),
                "labeled_precision_at_10": round(
                    sum(float(row["labeled_precision_at_10"]) for row in labeled_rows) / len(labeled_rows),
                    4,
                ) if labeled_rows else 0.0,
                "labeled_ndcg_at_10": round(
                    sum(float(row["labeled_ndcg_at_10"]) for row in labeled_rows) / len(labeled_rows),
                    4,
                ) if labeled_rows else 0.0,
            }
        )

    docker_stats = (
        {"available": False, "error": "Skipped by --skip-docker-stats"}
        if args.skip_docker_stats
        else capture_docker_stats(args.docker_container)
    )

    write_csv_rows(OUTPUT_DIR / "vector_optimization_summary.csv", summary_rows)
    write_csv_rows(OUTPUT_DIR / "vector_optimization_query_results.csv", query_rows)
    (OUTPUT_DIR / "artifact_sizes.json").write_text(json.dumps(artifact_sizes, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "docker_stats.json").write_text(json.dumps(docker_stats, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text(build_report(summary_rows, query_rows, docker_stats), encoding="utf-8")

    print(f"Generated vector optimization report: {OUTPUT_DIR / 'report.md'}")
    for row in summary_rows:
        print(
            f"- {row['variant']}: memory_500k={row['estimated_vector_memory_500k_mb']:.2f} MB, "
            f"reduction={row['memory_reduction_vs_fp32']:.2f}x, "
            f"overlap@10={row['avg_overlap_at_10']:.3f}, latency={row['avg_latency_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
