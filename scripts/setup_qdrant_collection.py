import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.search import QdrantStore
from scripts.qdrant_common import COLLECTION_NAME, QDRANT_PATH


def main() -> None:
    store = QdrantStore(collection_name=COLLECTION_NAME, qdrant_path=QDRANT_PATH)
    try:
        store.recreate_collection(vector_size=512)
    finally:
        store.close()

    print(f"Qdrant collection recreated: {COLLECTION_NAME}")
    print("Vector size: 512")
    print("Distance: Cosine")


if __name__ == "__main__":
    main()
