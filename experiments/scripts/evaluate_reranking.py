import csv
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder
from app.search import FaissIndex, MetadataRepository, RetrievalService, VectorStore
from experiments.paths import STYLE_RERANKING_DIR, ensure_experiment_directories


DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "flat.index"
METRICS_PATH = STYLE_RERANKING_DIR / "metrics.csv"
OUTPUT_PATH = STYLE_RERANKING_DIR / "evaluation_output.txt"
REPORT_PATH = STYLE_RERANKING_DIR / "report.md"
QUERY_SAMPLE_SIZE = 30
TOP_K = 10

COMPARISON_IMAGE_IDS = [
    "9U_uCvfpptk",
    "9wTWFyInJ4Y",
    "39DcBUbYZP4",
    "A-G8q9zorGs",
]


@dataclass
class MetricAccumulator:
    brightness_differences: list[float]
    contrast_differences: list[float]
    saturation_differences: list[float]
    warmth_differences: list[float]

    @classmethod
    def empty(cls) -> "MetricAccumulator":
        return cls([], [], [], [])

    def add_result(self, query_metadata, result_metadata) -> None:
        if query_metadata.brightness is not None and result_metadata.brightness is not None:
            self.brightness_differences.append(abs(query_metadata.brightness - result_metadata.brightness))
        if query_metadata.contrast is not None and result_metadata.contrast is not None:
            self.contrast_differences.append(abs(query_metadata.contrast - result_metadata.contrast))
        if query_metadata.saturation is not None and result_metadata.saturation is not None:
            self.saturation_differences.append(abs(query_metadata.saturation - result_metadata.saturation))
        if query_metadata.warmth is not None and result_metadata.warmth is not None:
            self.warmth_differences.append(abs(query_metadata.warmth - result_metadata.warmth))

    def averages(self) -> dict[str, float]:
        return {
            "brightness": statistics.fmean(self.brightness_differences) if self.brightness_differences else 0.0,
            "contrast": statistics.fmean(self.contrast_differences) if self.contrast_differences else 0.0,
            "saturation": statistics.fmean(self.saturation_differences) if self.saturation_differences else 0.0,
            "warmth": statistics.fmean(self.warmth_differences) if self.warmth_differences else 0.0,
        }


def create_service(rerank_enabled: bool) -> RetrievalService:
    vector_store = VectorStore(
        embeddings_path=EMBEDDINGS_PATH,
        image_ids_path=IMAGE_IDS_PATH,
    )
    faiss_index = FaissIndex(index_path=INDEX_PATH, dimension=512)
    faiss_index.load()
    metadata_repository = MetadataRepository(database_path=DATABASE_PATH)

    return RetrievalService(
        clip_encoder=CLIPEncoder(),
        vector_store=vector_store,
        faiss_index=faiss_index,
        metadata_repository=metadata_repository,
        rerank_enabled=rerank_enabled,
        candidate_pool_size=100,
    )


def load_query_rows(limit: int) -> list[tuple[str, str]]:
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT image_id, file_path
            FROM images
            WHERE brightness IS NOT NULL
              AND contrast IS NOT NULL
              AND saturation IS NOT NULL
              AND warmth IS NOT NULL
              AND color_histogram IS NOT NULL
            ORDER BY image_id
            LIMIT ?
            """,
            (limit,),
        )
        return cursor.fetchall()
    finally:
        conn.close()


def resolve_path(file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate_service(service: RetrievalService, query_rows: list[tuple[str, str]]) -> MetricAccumulator:
    metrics = MetricAccumulator.empty()

    for image_id, file_path in query_rows:
        query_path = resolve_path(file_path)
        query_metadata = service.metadata_repository.get_by_id(image_id)
        if query_metadata is None:
            continue

        results = service.search_by_image(query_path, top_k=TOP_K + 1)
        filtered_results = [result for result in results if result["image_id"] != image_id][:TOP_K]

        result_ids = [str(result["image_id"]) for result in filtered_results]
        result_metadata_map = service.metadata_repository.get_many(result_ids)

        for result_id in result_ids:
            result_metadata = result_metadata_map.get(result_id)
            if result_metadata is None:
                continue
            metrics.add_result(query_metadata, result_metadata)

    return metrics


def build_console_lines(
    baseline: dict[str, float],
    reranked: dict[str, float],
    query_count: int,
) -> list[str]:
    lines = [
        f"Evaluating reranking on {query_count} query images",
        "",
        "CLIP only:",
        f"avg brightness difference = {baseline['brightness']:.4f}",
        f"avg contrast difference = {baseline['contrast']:.4f}",
        f"avg saturation difference = {baseline['saturation']:.4f}",
        f"avg warmth difference = {baseline['warmth']:.4f}",
        "",
        "Reranked:",
        f"avg brightness difference = {reranked['brightness']:.4f}",
        f"avg contrast difference = {reranked['contrast']:.4f}",
        f"avg saturation difference = {reranked['saturation']:.4f}",
        f"avg warmth difference = {reranked['warmth']:.4f}",
        "",
        "Improvement (positive means reranked is closer in style):",
    ]
    for metric_name in ("brightness", "contrast", "saturation", "warmth"):
        improvement = baseline[metric_name] - reranked[metric_name]
        lines.append(f"{metric_name}: {improvement:.4f}")
    return lines


def write_metrics_csv(
    baseline: dict[str, float],
    reranked: dict[str, float],
) -> None:
    with METRICS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "clip_only", "reranked", "improvement"],
        )
        writer.writeheader()
        for metric_name in ("brightness", "contrast", "saturation", "warmth"):
            writer.writerow(
                {
                    "metric": metric_name,
                    "clip_only": f"{baseline[metric_name]:.6f}",
                    "reranked": f"{reranked[metric_name]:.6f}",
                    "improvement": f"{baseline[metric_name] - reranked[metric_name]:.6f}",
                }
            )


def build_report(
    baseline: dict[str, float],
    reranked: dict[str, float],
    query_count: int,
) -> str:
    comparison_lines: list[str] = []
    for image_id in COMPARISON_IMAGE_IDS:
        comparison_lines.extend(
            [
                f"## Query Image: {image_id}",
                "",
                f"![](visualizations/compare_{image_id}.jpg)",
                "",
            ]
        )
    metrics_rows = "\n".join(
        f"| {metric_name} | {baseline[metric_name]:.4f} | {reranked[metric_name]:.4f} | {baseline[metric_name] - reranked[metric_name]:.4f} |"
        for metric_name in ("brightness", "contrast", "saturation", "warmth")
    )
    return "\n".join(
        [
            "# Style Reranking Report",
            "",
            "This stage measures whether style-aware reranking produces neighbors that are visually closer to the query than the FAISS semantic baseline alone.",
            "",
            f"- Query sample size: {query_count}",
            f"- Candidate pool size before reranking: 100",
            f"- Evaluated top-k per query: {TOP_K}",
            "",
            "## Metrics",
            "",
            "| Metric | CLIP only | Reranked | Improvement |",
            "| --- | ---: | ---: | ---: |",
            metrics_rows,
            "",
            "Improvement is baseline difference minus reranked difference, so positive values mean the reranker moved the results closer to the query style.",
            "",
            "## Visual Comparisons",
            "",
            *comparison_lines,
            "",
            "## Reproduce",
            "",
            "```bash",
            "python experiments/scripts/evaluate_style_reranking.py",
            "```",
        ]
    )


def main() -> None:
    ensure_experiment_directories()
    query_rows = load_query_rows(QUERY_SAMPLE_SIZE)
    if not query_rows:
        raise RuntimeError("No query rows available for reranking evaluation")

    clip_only_service = create_service(rerank_enabled=False)
    reranked_service = create_service(rerank_enabled=True)

    clip_only_averages = evaluate_service(clip_only_service, query_rows).averages()
    reranked_averages = evaluate_service(reranked_service, query_rows).averages()

    console_lines = build_console_lines(
        baseline=clip_only_averages,
        reranked=reranked_averages,
        query_count=len(query_rows),
    )
    console_output = "\n".join(console_lines) + "\n"

    print(console_output, end="")
    OUTPUT_PATH.write_text(console_output, encoding="utf-8")
    write_metrics_csv(clip_only_averages, reranked_averages)
    REPORT_PATH.write_text(
        build_report(clip_only_averages, reranked_averages, len(query_rows)),
        encoding="utf-8",
    )

    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved evaluation log: {OUTPUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
