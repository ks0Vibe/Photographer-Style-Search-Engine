import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.search import FaissIndex, VectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS Flat index from saved embeddings.")
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy",
        help="Path to the embeddings .npy file.",
    )
    parser.add_argument(
        "--image-ids-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy",
        help="Path to the image IDs .npy file.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "indexes" / "flat.index",
        help="Path to the output FAISS index file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vector_store = VectorStore(
        embeddings_path=args.embeddings_path,
        image_ids_path=args.image_ids_path,
    )
    embeddings = vector_store.get_embeddings()
    vector_store.get_image_ids()

    print(f"Embeddings loaded: {embeddings.shape[0]}")
    print(f"Dimension: {embeddings.shape[1]}")
    print()
    print("Building FAISS Flat index...")
    print()

    faiss_index = FaissIndex(
        index_path=args.index_path,
        dimension=embeddings.shape[1],
    )
    faiss_index.build(embeddings)
    faiss_index.save()

    print(f"Indexed vectors: {embeddings.shape[0]}")
    print()
    print("Saved:")
    try:
        display_path = args.index_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = args.index_path
    print(display_path)


if __name__ == "__main__":
    main()
