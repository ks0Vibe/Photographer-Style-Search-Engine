import csv
import sqlite3
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder
from app.search import (
    FaissIndex,
    MetadataRepository,
    QdrantRetrievalService,
    RetrievalService,
    VectorStore,
)
from experiments.paths import QDRANT_BACKEND_DIR, ensure_experiment_directories
from scripts.qdrant_common import DATABASE_PATH, EMBEDDINGS_PATH, IMAGE_IDS_PATH, QDRANT_PATH, create_qdrant_store, resolve_image_path


FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "flat.index"
CSV_OUTPUT_PATH = QDRANT_BACKEND_DIR / "faiss_vs_qdrant_results.csv"
MD_OUTPUT_PATH = QDRANT_BACKEND_DIR / "faiss_vs_qdrant_results.md"
REPORT_PATH = QDRANT_BACKEND_DIR / "report.md"
TEXT_QUERIES = [
    "warm cinematic landscape",
    "dark moody forest",
    "street photography",
    "portrait",
    "beach sunset",
]
IMAGE_QUERY_COUNT = 30
TOP_K = 10


def build_faiss_service(clip_encoder: CLIPEncoder) -> RetrievalService:
    vector_store = VectorStore(
        embeddings_path=EMBEDDINGS_PATH,
        image_ids_path=IMAGE_IDS_PATH,
    )
    metadata_repository = MetadataRepository(DATABASE_PATH)
    faiss_index = FaissIndex(index_path=FAISS_INDEX_PATH, dimension=512)
    faiss_index.load()
    return RetrievalService(
        clip_encoder=clip_encoder,
        vector_store=vector_store,
        faiss_index=faiss_index,
        metadata_repository=metadata_repository,
    )


def build_qdrant_service(clip_encoder: CLIPEncoder) -> QdrantRetrievalService:
    return QdrantRetrievalService(
        clip_encoder=clip_encoder,
        qdrant_store=create_qdrant_store(),
    )


def load_image_queries(limit: int) -> list[tuple[str, Path]]:
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT image_id, file_path
            FROM images
            ORDER BY image_id
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [(str(image_id), resolve_image_path(str(file_path))) for image_id, file_path in rows]


def timed_call(fn):
    started = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, elapsed_ms


def overlap_at_k(left_results: list[dict], right_results: list[dict], k: int) -> float:
    left_ids = {str(result["image_id"]) for result in left_results[:k]}
    right_ids = {str(result["image_id"]) for result in right_results[:k]}
    if k <= 0:
        return 0.0
    return len(left_ids & right_ids) / k


def top1_match(left_results: list[dict], right_results: list[dict]) -> int:
    if not left_results or not right_results:
        return 0
    return int(str(left_results[0]["image_id"]) == str(right_results[0]["image_id"]))


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_type",
                "query_label",
                "faiss_latency_ms",
                "qdrant_latency_ms",
                "overlap_at_10",
                "top1_match",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, round(0.95 * (len(ordered) - 1)))
    return ordered[index]


def build_markdown(rows: list[dict[str, object]]) -> str:
    faiss_latencies = [float(row["faiss_latency_ms"]) for row in rows]
    qdrant_latencies = [float(row["qdrant_latency_ms"]) for row in rows]
    overlaps = [float(row["overlap_at_10"]) for row in rows]
    consistency = [int(row["top1_match"]) for row in rows]

    faiss_size = FAISS_INDEX_PATH.stat().st_size if FAISS_INDEX_PATH.exists() else 0
    qdrant_size = directory_size(QDRANT_PATH)

    return "\n".join(
        [
            "# FAISS vs Qdrant Comparison",
            "",
            "This stage compares the baseline FAISS backend against the Qdrant backend under unfiltered retrieval so the advanced backend can be judged against a stable exact-search reference.",
            "",
            "## Setup",
            "",
            "- Query set: 30 image queries and 5 text queries",
            f"- Top-K: {TOP_K}",
            "- FAISS mode: Flat cosine search",
            "- Qdrant mode: local persistent cosine collection with payloads",
            "",
            "## Results",
            "",
            f"- FAISS average latency: {statistics.fmean(faiss_latencies):.2f} ms",
            f"- FAISS p95 latency: {p95(faiss_latencies):.2f} ms",
            f"- Qdrant average latency: {statistics.fmean(qdrant_latencies):.2f} ms",
            f"- Qdrant p95 latency: {p95(qdrant_latencies):.2f} ms",
            f"- Average overlap@10: {statistics.fmean(overlaps):.3f}",
            f"- Top-1 consistency: {statistics.fmean(consistency):.3f}",
            f"- FAISS index size: {faiss_size} bytes",
            f"- Qdrant local data size: {qdrant_size} bytes",
            "",
            "## Interpretation",
            "",
            "- FAISS stays as the exact local baseline.",
            "- Qdrant remains close enough to baseline behavior for unfiltered search while adding database features that FAISS does not provide.",
            "- The visualizations in `visualizations/` show sample Qdrant retrieval outputs produced by the same backend.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python experiments/scripts/compare_faiss_vs_qdrant.py",
            "```",
        ]
    )


def main() -> None:
    ensure_experiment_directories()
    clip_encoder = CLIPEncoder()
    faiss_service = build_faiss_service(clip_encoder)
    qdrant_service = build_qdrant_service(clip_encoder)

    rows: list[dict[str, object]] = []
    try:
        for image_id, image_path in load_image_queries(IMAGE_QUERY_COUNT):
            faiss_results, faiss_latency_ms = timed_call(
                lambda: faiss_service.search_by_image(image_path=image_path, top_k=TOP_K + 1, rerank_enabled=False)
            )
            qdrant_results, qdrant_latency_ms = timed_call(
                lambda: qdrant_service.search_by_image(image_path=image_path, top_k=TOP_K + 1, rerank=False)
            )

            faiss_results = [result for result in faiss_results if result["image_id"] != image_id][:TOP_K]
            qdrant_results = [result for result in qdrant_results if result["image_id"] != image_id][:TOP_K]

            rows.append(
                {
                    "query_type": "image",
                    "query_label": image_id,
                    "faiss_latency_ms": round(faiss_latency_ms, 4),
                    "qdrant_latency_ms": round(qdrant_latency_ms, 4),
                    "overlap_at_10": round(overlap_at_k(faiss_results, qdrant_results, TOP_K), 4),
                    "top1_match": top1_match(faiss_results, qdrant_results),
                }
            )

        for query in TEXT_QUERIES:
            faiss_results, faiss_latency_ms = timed_call(
                lambda: faiss_service.search_by_text(text=query, top_k=TOP_K, rerank_enabled=False)
            )
            qdrant_results, qdrant_latency_ms = timed_call(
                lambda: qdrant_service.search_by_text(text=query, top_k=TOP_K, rerank=False)
            )

            rows.append(
                {
                    "query_type": "text",
                    "query_label": query,
                    "faiss_latency_ms": round(faiss_latency_ms, 4),
                    "qdrant_latency_ms": round(qdrant_latency_ms, 4),
                    "overlap_at_10": round(overlap_at_k(faiss_results, qdrant_results, TOP_K), 4),
                    "top1_match": top1_match(faiss_results, qdrant_results),
                }
            )
    finally:
        qdrant_service.close()

    write_csv(rows)
    markdown = build_markdown(rows)
    MD_OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    REPORT_PATH.write_text(markdown, encoding="utf-8")

    print(f"Saved CSV: {CSV_OUTPUT_PATH}")
    print(f"Saved report: {MD_OUTPUT_PATH}")
    print(f"Saved stage report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
