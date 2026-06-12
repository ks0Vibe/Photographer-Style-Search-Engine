import argparse
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder
from app.search import FaissIndex, MetadataRepository, RetrievalService, VectorStore


DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "flat.index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an image-to-image retrieval smoke test.")
    parser.add_argument("--image-id", type=str, default=None, help="Image ID to use as the query.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to display.")
    return parser.parse_args()


def get_query_row(image_id: str | None) -> tuple[str, str]:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        if image_id:
            cursor.execute(
                """
                SELECT image_id, file_path
                FROM images
                WHERE image_id = ?
                """,
                (image_id,),
            )
        else:
            cursor.execute(
                """
                SELECT image_id, file_path
                FROM images
                ORDER BY image_id
                LIMIT 1
                """
            )

        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None:
        label = image_id if image_id else "deterministic query row"
        raise RuntimeError(f"Could not find {label} in the metadata database")

    return row[0], row[1]


def build_service() -> RetrievalService:
    vector_store = VectorStore(
        embeddings_path=EMBEDDINGS_PATH,
        image_ids_path=IMAGE_IDS_PATH,
    )
    metadata_repository = MetadataRepository(DATABASE_PATH)
    faiss_index = FaissIndex(index_path=INDEX_PATH)
    faiss_index.load()

    return RetrievalService(
        clip_encoder=CLIPEncoder(),
        vector_store=vector_store,
        faiss_index=faiss_index,
        metadata_repository=metadata_repository,
    )


def resolve_image_path(file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    query_image_id, file_path = get_query_row(args.image_id)
    query_image_path = resolve_image_path(file_path)
    service = build_service()

    requested_top_k = args.top_k + 1 if args.image_id is None else args.top_k
    results = service.search_by_image(query_image_path, top_k=requested_top_k)

    if args.image_id is None and results and results[0]["image_id"] == query_image_id:
        results = [result for result in results if result["image_id"] != query_image_id]

    results = results[: args.top_k]

    print("Query:")
    print(query_image_id)
    print()
    print("Results:")
    print()

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['image_id']} score={result['score']:.4f}")
        print(f"   file_path={result['file_path']}")


if __name__ == "__main__":
    main()
