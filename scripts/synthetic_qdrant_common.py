from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.search import QdrantStore


SYNTHETIC_COLLECTION_NAME = "photos_synthetic_500k"
SYNTHETIC_QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_synthetic_500k"
SERVER_QDRANT_STORAGE_PATH = PROJECT_ROOT / "data" / "qdrant_storage"
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic_500k"
SYNTHETIC_STAGE_DIR = PROJECT_ROOT / "experiments" / "07_synthetic_500k_scale"


def synthetic_collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION", SYNTHETIC_COLLECTION_NAME).strip() or SYNTHETIC_COLLECTION_NAME


def synthetic_qdrant_mode() -> str:
    mode = os.getenv("QDRANT_MODE", "local").strip().lower()
    if mode not in {"local", "server"}:
        raise ValueError("QDRANT_MODE must be either 'local' or 'server'")
    return mode


def synthetic_qdrant_url() -> str:
    return os.getenv("QDRANT_URL", "http://localhost:6333").strip()


def synthetic_qdrant_path() -> Path:
    raw_path = os.getenv("QDRANT_PATH", str(SYNTHETIC_QDRANT_PATH))
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_synthetic_qdrant_store() -> QdrantStore:
    mode = synthetic_qdrant_mode()
    kwargs: dict[str, Any] = {"collection_name": synthetic_collection_name()}
    if mode == "server":
        kwargs["qdrant_url"] = synthetic_qdrant_url()
    else:
        kwargs["qdrant_path"] = synthetic_qdrant_path()
    return QdrantStore(**kwargs)


def synthetic_storage_label() -> str:
    if synthetic_qdrant_mode() == "server":
        return synthetic_qdrant_url()
    return str(synthetic_qdrant_path())


def directory_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total_bytes = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_bytes += file_path.stat().st_size
    return total_bytes / (1024 * 1024)
