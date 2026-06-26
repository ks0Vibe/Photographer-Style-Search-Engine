from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class QdrantSettings:
    mode: str
    collection_name: str
    path: Path
    url: str

    @property
    def storage_label(self) -> str:
        if self.mode == "server":
            return self.url
        return str(self.path)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_qdrant_settings() -> QdrantSettings:
    mode = os.getenv("QDRANT_MODE", "local").strip().lower()
    if mode not in {"local", "server"}:
        raise ValueError("QDRANT_MODE must be either 'local' or 'server'")

    return QdrantSettings(
        mode=mode,
        collection_name=os.getenv("QDRANT_COLLECTION", "photos").strip() or "photos",
        path=resolve_project_path(os.getenv("QDRANT_PATH", "data/qdrant")),
        url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
    )


def create_qdrant_client(settings: QdrantSettings | None = None) -> QdrantClient:
    resolved = settings or get_qdrant_settings()
    if resolved.mode == "server":
        return QdrantClient(url=resolved.url)

    resolved.path.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(resolved.path))
