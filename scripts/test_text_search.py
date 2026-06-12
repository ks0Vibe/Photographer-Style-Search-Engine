import argparse
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
    parser = argparse.ArgumentParser(description="Run a text-to-image retrieval smoke test.")
    parser.add_argument(
        "--query",
        type=str,
        default="warm cinematic landscape",
        help="Text query to search for.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to display.")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    service = build_service()
    results = service.search_by_text(args.query, top_k=args.top_k)

    print("Query:")
    print(args.query)
    print()
    print("Results:")
    print()

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. image_id={result['image_id']} "
            f"score={result['score']:.4f} file_path={result['file_path']}"
        )


if __name__ == "__main__":
    main()
