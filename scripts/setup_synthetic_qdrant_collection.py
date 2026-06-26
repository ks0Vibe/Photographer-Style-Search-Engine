from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.synthetic_qdrant_common import (
    create_synthetic_qdrant_store,
    synthetic_collection_name,
    synthetic_qdrant_mode,
    synthetic_storage_label,
)


def main() -> None:
    store = create_synthetic_qdrant_store()
    try:
        store.recreate_collection(vector_size=512)
    finally:
        store.close()

    print(f"Qdrant collection recreated: {synthetic_collection_name()}")
    print(f"Mode: {synthetic_qdrant_mode()}")
    print(f"Storage: {synthetic_storage_label()}")
    print("Vector size: 512")
    print("Distance: Cosine")
    if synthetic_qdrant_mode() == "local":
        print("Note: Docker/server Qdrant is preferred for the 500k synthetic benchmark.")


if __name__ == "__main__":
    main()
