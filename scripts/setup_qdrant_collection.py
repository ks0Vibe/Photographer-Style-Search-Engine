import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qdrant_common import COLLECTION_NAME, QDRANT_MODE, create_qdrant_store, qdrant_storage_label


def main() -> None:
    store = create_qdrant_store()
    try:
        store.recreate_collection(vector_size=512)
    finally:
        store.close()

    print(f"Qdrant collection recreated: {COLLECTION_NAME}")
    print(f"Mode: {QDRANT_MODE}")
    print(f"Storage: {qdrant_storage_label()}")
    print("Vector size: 512")
    print("Distance: Cosine")


if __name__ == "__main__":
    main()
