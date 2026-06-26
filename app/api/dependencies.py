from __future__ import annotations

import json
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.clip_encoder import CLIPEncoder
from app.search import QdrantRetrievalService, QdrantStore, load_keywords_by_image_id
from app.search.qdrant_config import PROJECT_ROOT, get_qdrant_settings


DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
KEYWORDS_PATH = PROJECT_ROOT / "data" / "unsplash-lite" / "keywords.csv000"


def normalize_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip().lower() for item in value if str(item).strip()})


class MetadataLookup:
    def __init__(self, database_path: Path = DATABASE_PATH, keywords_path: Path = KEYWORDS_PATH) -> None:
        self.database_path = Path(database_path)
        self.keywords_path = Path(keywords_path)

    def _connect(self) -> sqlite3.Connection:
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    @lru_cache(maxsize=1)
    def keywords_by_image_id(self) -> dict[str, list[str]]:
        if not self.keywords_path.exists():
            return {}
        return load_keywords_by_image_id(self.keywords_path)

    def get_image(self, image_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(images);").fetchall()}
            detected_select = "detected_objects" if "detected_objects" in columns else "'[]' AS detected_objects"
            row = conn.execute(
                f"""
                SELECT
                    image_id,
                    file_path,
                    photo_url,
                    ai_description,
                    photographer_username,
                    brightness,
                    contrast,
                    saturation,
                    warmth,
                    {detected_select}
                FROM images
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["keywords"] = self.keywords_by_image_id().get(str(image_id), [])
        data["detected_objects"] = normalize_json_list(data.get("detected_objects"))
        return data

    def get_many(self, image_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {
            image_id: metadata
            for image_id in image_ids
            if (metadata := self.get_image(image_id)) is not None
        }

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(images);").fetchall()}
            total = int(conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])
            if "detected_objects" in columns:
                with_objects = int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM images
                        WHERE detected_objects IS NOT NULL
                          AND detected_objects != ''
                          AND detected_objects != '[]'
                        """
                    ).fetchone()[0]
                )
            else:
                with_objects = 0
        return {
            "sqlite_image_rows": total,
            "images_with_detected_objects": with_objects,
            "object_coverage": with_objects / total if total else 0.0,
        }

    def resolve_image_path(self, image_id: str) -> Path | None:
        metadata = self.get_image(image_id)
        if metadata is None:
            return None
        raw_path = Path(str(metadata.get("file_path") or ""))
        resolved = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        resolved = resolved.resolve()
        project_root = PROJECT_ROOT.resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError:
            return None
        return resolved if resolved.exists() and resolved.is_file() else None


class SearchApplication:
    def __init__(self) -> None:
        settings = get_qdrant_settings()
        store_kwargs: dict[str, Any] = {"collection_name": settings.collection_name}
        if settings.mode == "server":
            store_kwargs["qdrant_url"] = settings.url
        else:
            store_kwargs["qdrant_path"] = settings.path

        self.settings = settings
        self.metadata = MetadataLookup()
        self.service = QdrantRetrievalService(
            clip_encoder=CLIPEncoder(),
            qdrant_store=QdrantStore(**store_kwargs),
        )

    def close(self) -> None:
        self.service.close()

    def qdrant_points(self) -> int:
        return self.service.qdrant_store.count()


_APPLICATION_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _build_search_application() -> SearchApplication:
    return SearchApplication()


def get_search_application() -> SearchApplication:
    with _APPLICATION_LOCK:
        return _build_search_application()


def search_application_cache_info():
    return _build_search_application.cache_info()


def clear_search_application_cache() -> None:
    _build_search_application.cache_clear()


def get_embedding_dim() -> int | None:
    if not EMBEDDINGS_PATH.exists():
        return None
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    if embeddings.ndim != 2:
        return None
    return int(embeddings.shape[1])
