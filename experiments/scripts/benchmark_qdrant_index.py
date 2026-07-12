from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.synthetic_qdrant_common import (  # noqa: E402
    SERVER_QDRANT_STORAGE_PATH,
    SYNTHETIC_DATA_DIR,
    synthetic_qdrant_mode,
    synthetic_qdrant_path,
    synthetic_qdrant_url,
)


OUTPUT_DIR = PROJECT_ROOT / "experiments" / "13_index_selection"
RESULTS_PATH = OUTPUT_DIR / "qdrant_index_benchmark.csv"
REPORT_PATH = OUTPUT_DIR / "qdrant_index_benchmark.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact FAISS ground truth with explicit Qdrant HNSW and native scalar INT8."
    )
    parser.add_argument("--data-dir", type=Path, default=SYNTHETIC_DATA_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-count", type=int, default=30)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--container", default="photographer-style-qdrant")
    parser.add_argument("--collection-prefix", default="index_benchmark")
    parser.add_argument(
        "--hnsw-config",
        action="append",
        default=None,
        metavar="M,EF_CONSTRUCT,EF_SEARCH",
        help="Repeat for each HNSW variant. Default: 16,100,64; 32,200,128; 64,400,256.",
    )
    parser.add_argument(
        "--skip-scalar-int8",
        action="store_true",
        help="Do not run the native Qdrant scalar INT8 variant.",
    )
    parser.add_argument(
        "--always-ram",
        action="store_true",
        help="Keep native scalar quantized vectors in RAM instead of allowing mmap.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / (1024 * 1024)


def storage_path() -> Path:
    return synthetic_qdrant_path() if synthetic_qdrant_mode() == "local" else SERVER_QDRANT_STORAGE_PATH


def qdrant_storage_size_mb(container: str, collection: str | None = None) -> float:
    if synthetic_qdrant_mode() != "server":
        return directory_size_mb(storage_path())
    try:
        target = "/qdrant/storage/collections"
        if collection:
            target += f"/{collection}"
        result = subprocess.run(
            ["docker", "exec", container, "sh", "-c", f"du -sb {target} | cut -f1"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip()) / (1024 * 1024)
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        if collection and synthetic_qdrant_mode() == "server":
            return 0.0
        return directory_size_mb(storage_path())


def parse_memory_mb(raw: str) -> float | None:
    match = re.match(r"\s*([0-9.]+)\s*([KMGT]?i?B)", raw or "", re.IGNORECASE)
    if not match:
        return None
    units = {"b": 1, "kb": 1024, "kib": 1024, "mb": 1024**2, "mib": 1024**2,
             "gb": 1024**3, "gib": 1024**3, "tb": 1024**4, "tib": 1024**4}
    return float(match.group(1)) * units[match.group(2).lower()] / (1024 * 1024)


def docker_stats(container: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", container],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
        )
        raw = result.stdout.strip().splitlines()[0]
        parsed = json.loads(raw)
        parsed["available"] = True
        parsed["memory_usage_mb"] = parse_memory_mb(str(parsed.get("MemUsage", "")).split("/")[0])
        parsed["memory_limit_mb"] = parse_memory_mb(str(parsed.get("MemUsage", "")).split("/")[-1])
        return parsed
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, json.JSONDecodeError) as exc:
        return {"available": False, "error": str(exc)}


def client() -> QdrantClient:
    if synthetic_qdrant_mode() == "server":
        return QdrantClient(url=synthetic_qdrant_url())
    return QdrantClient(path=str(synthetic_qdrant_path()))


def parse_hnsw_configs(values: list[str] | None) -> list[tuple[int, int, int]]:
    raw_values = values or ["16,100,64", "32,200,128", "64,400,256"]
    configs = []
    for raw in raw_values:
        parts = [int(part.strip()) for part in raw.split(",")]
        if len(parts) != 3 or min(parts) <= 0:
            raise ValueError(f"Invalid --hnsw-config {raw!r}; expected M,EF_CONSTRUCT,EF_SEARCH")
        configs.append((parts[0], parts[1], parts[2]))
    return configs


def load_data(data_dir: Path, query_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.load(data_dir / "synthetic_embeddings.npy", mmap_mode="r", allow_pickle=False)
    image_ids = np.load(data_dir / "synthetic_image_ids.npy", mmap_mode="r", allow_pickle=False)
    real_embeddings = np.load(PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Expected synthetic (N, 512), got {embeddings.shape}")
    if query_count > len(real_embeddings):
        raise ValueError("--query-count is greater than the real embedding corpus")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(real_embeddings), size=query_count, replace=False)
    queries = np.asarray(real_embeddings[indices], dtype=np.float32)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    return np.asarray(embeddings, dtype=np.float32), np.asarray(image_ids), queries


def exact_ground_truth(embeddings: np.ndarray, image_ids: np.ndarray, queries: np.ndarray, top_k: int) -> list[set[str]]:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, indices = index.search(queries, top_k)
    return [{str(image_ids[i]) for i in row if i >= 0} for row in indices]


def wait_until_green(qdrant: QdrantClient, collection: str, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = str(qdrant.get_collection(collection).status).lower()
        if "green" in status:
            return
        time.sleep(2)
    raise TimeoutError(f"Qdrant collection did not become green: {collection}")


def create_collection(qdrant: QdrantClient, collection: str, args: argparse.Namespace, *, m: int, ef_construct: int, scalar: bool) -> None:
    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)
    quantization = None
    if scalar:
        quantization = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,
                always_ram=args.always_ram,
            )
        )
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
        quantization_config=quantization,
        on_disk_payload=True,
    )


def clear_benchmark_collections(qdrant: QdrantClient, collection_prefix: str) -> None:
    for collection_info in qdrant.get_collections().collections:
        if str(collection_info.name).startswith(f"{collection_prefix}_"):
            qdrant.delete_collection(collection_info.name)
    time.sleep(3)


def upload(qdrant: QdrantClient, collection: str, embeddings: np.ndarray, image_ids: np.ndarray, batch_size: int) -> None:
    for start in range(0, len(embeddings), batch_size):
        end = min(start + batch_size, len(embeddings))
        points = [
            models.PointStruct(id=i, vector=embeddings[i].tolist(), payload={"image_id": str(image_ids[i])})
            for i in range(start, end)
        ]
        qdrant.upsert(collection_name=collection, points=points, wait=True)


def run_searches(qdrant: QdrantClient, collection: str, queries: np.ndarray, ground_truth: list[set[str]], top_k: int, ef_search: int, latency_runs: int) -> tuple[float, float, float]:
    latencies: list[float] = []
    recalls: list[float] = []
    for query_index, query in enumerate(queries):
        response = qdrant.query_points(
            collection_name=collection,
            query=query.tolist(),
            limit=top_k,
            search_params=models.SearchParams(hnsw_ef=ef_search, exact=False),
            with_payload=True,
            with_vectors=False,
        )
        returned = {str(point.payload.get("image_id")) for point in response.points if point.payload}
        recalls.append(len(returned & ground_truth[query_index]) / top_k)
        for _ in range(latency_runs):
            started = time.perf_counter()
            qdrant.query_points(
                collection_name=collection,
                query=query.tolist(),
                limit=top_k,
                search_params=models.SearchParams(hnsw_ef=ef_search, exact=False),
                with_payload=False,
                with_vectors=False,
            )
            latencies.append((time.perf_counter() - started) * 1000)
    return statistics.fmean(recalls), percentile(latencies, 0.50), percentile(latencies, 0.95)


def write_report(rows: list[dict[str, Any]], exact_build_seconds: float, corpus_size: int) -> None:
    columns = ["variant", "collection_size", "m", "ef_construct", "ef_search", "native_scalar_int8", "build_time_seconds", "recall_at_10", "p50_latency_ms", "p95_latency_ms", "disk_after_mb", "disk_delta_mb", "container_memory_mb"]
    lines = [
        "# Qdrant Index Selection Benchmark",
        "",
        "This benchmark uses an exact FAISS `IndexFlatIP` over the same synthetic 500k vectors as ground truth. Every Qdrant variant has explicit HNSW settings and is queried with the same normalized query vectors.",
        "",
        f"- Corpus: `{corpus_size}` vectors, 512 dimensions",
        f"- Exact FAISS ground-truth build time: `{exact_build_seconds:.3f}` s",
        "- Recall@10 is against the exact FAISS top-10 set; p50/p95 are Qdrant-only search latency after one warm-up query per query vector.",
        "- `native_scalar_int8=true` means Qdrant scalar quantization is configured server-side; no Python dequantization is included.",
        "",
        "## Results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "Select the lowest-latency configuration whose Recall@10 meets the project threshold (recommended >= 0.98). If native INT8 meets that threshold, it is the production-like memory winner; otherwise select the HNSW setting on the Pareto frontier and keep INT8 as a measured alternative.",
        "",
        "## Reproduce",
        "",
        "```powershell",
        "$env:QDRANT_MODE='server'",
        "$env:QDRANT_URL='http://localhost:6333'",
        ".\\.venv\\Scripts\\python.exe experiments\\scripts\\benchmark_qdrant_index.py",
        "```",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_k != 10:
        print("Warning: output column is named recall_at_10; use --top-k 10 for the requested metric.", flush=True)
    if args.top_k <= 0 or args.query_count <= 0 or args.latency_runs <= 0:
        raise ValueError("top-k, query-count and latency-runs must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_path(args.data_dir)
    embeddings, image_ids, queries = load_data(data_dir, args.query_count, args.seed)
    gt_started = time.perf_counter()
    ground_truth = exact_ground_truth(embeddings, image_ids, queries, args.top_k)
    exact_build_seconds = time.perf_counter() - gt_started
    qdrant = client()
    rows: list[dict[str, Any]] = []
    try:
        variants: list[tuple[str, int, int, int, bool]] = [
            (f"hnsw_m{m}_efc{efc}_efs{efs}", m, efc, efs, False)
            for m, efc, efs in parse_hnsw_configs(args.hnsw_config)
        ]
        if not args.skip_scalar_int8:
            variants.append(("native_scalar_int8_m32_efc200_efs128", 32, 200, 128, True))
        for variant, m, ef_construct, ef_search, scalar in variants:
            collection = f"{args.collection_prefix}_{variant}"[:200]
            clear_benchmark_collections(qdrant, args.collection_prefix)
            disk_before = qdrant_storage_size_mb(args.container, collection)
            started = time.perf_counter()
            create_collection(qdrant, collection, args, m=m, ef_construct=ef_construct, scalar=scalar)
            upload(qdrant, collection, embeddings, image_ids, args.batch_size)
            wait_until_green(qdrant, collection)
            build_seconds = time.perf_counter() - started
            disk_after = qdrant_storage_size_mb(args.container, collection)
            recall, p50, p95 = run_searches(qdrant, collection, queries, ground_truth, args.top_k, ef_search, args.latency_runs)
            stats = docker_stats(args.container)
            rows.append({
                "variant": variant,
                "collection": collection,
                "collection_size": len(embeddings),
                "m": m,
                "ef_construct": ef_construct,
                "ef_search": ef_search,
                "native_scalar_int8": scalar,
                "build_time_seconds": round(build_seconds, 3),
                "recall_at_10": round(recall, 4),
                "p50_latency_ms": round(p50, 3),
                "p95_latency_ms": round(p95, 3),
                "disk_after_mb": round(disk_after, 3),
                "disk_delta_mb": round(max(0.0, disk_after - disk_before), 3),
                "container_memory_mb": stats.get("memory_usage_mb", ""),
                "container_mem_usage": stats.get("MemUsage", ""),
                "container_cpu": stats.get("CPUPerc", ""),
                "docker_stats_available": stats.get("available", False),
            })
            print(f"{variant}: recall@10={recall:.4f}, p50={p50:.2f} ms, p95={p95:.2f} ms, build={build_seconds:.1f} s", flush=True)
    finally:
        qdrant.close()
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows, exact_build_seconds, len(embeddings))
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
