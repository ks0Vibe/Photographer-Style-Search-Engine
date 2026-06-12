import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder
from app.search import FaissIndex, MetadataRepository, RetrievalService, VectorStore


DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "flat.index"
QUERY_SAMPLE_SIZE = 30
TOP_K = 10


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


def print_report(title: str, averages: dict[str, float]) -> None:
    print(f"{title}:")
    print(f"avg brightness difference = {averages['brightness']:.4f}")
    print(f"avg contrast difference = {averages['contrast']:.4f}")
    print(f"avg saturation difference = {averages['saturation']:.4f}")
    print(f"avg warmth difference = {averages['warmth']:.4f}")
    print()


def print_improvements(
    baseline: dict[str, float],
    reranked: dict[str, float],
) -> None:
    print("Improvement (positive means reranked is closer in style):")
    for metric_name in ("brightness", "contrast", "saturation", "warmth"):
        improvement = baseline[metric_name] - reranked[metric_name]
        print(f"{metric_name}: {improvement:.4f}")


def main() -> None:
    query_rows = load_query_rows(QUERY_SAMPLE_SIZE)
    if not query_rows:
        raise RuntimeError("No query rows available for reranking evaluation")

    print(f"Evaluating reranking on {len(query_rows)} query images")
    print()

    clip_only_service = create_service(rerank_enabled=False)
    reranked_service = create_service(rerank_enabled=True)

    clip_only_averages = evaluate_service(clip_only_service, query_rows).averages()
    reranked_averages = evaluate_service(reranked_service, query_rows).averages()

    print_report("CLIP only", clip_only_averages)
    print_report("Reranked", reranked_averages)
    print_improvements(clip_only_averages, reranked_averages)


if __name__ == "__main__":
    main()
