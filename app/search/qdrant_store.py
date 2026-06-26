from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import models

from app.search.qdrant_config import QdrantSettings, create_qdrant_client


logger = logging.getLogger(__name__)


STYLE_FIELDS = ("brightness", "contrast", "saturation", "warmth")


def load_keywords_by_image_id(path: Path) -> dict[str, list[str]]:
    keywords_path = Path(path)
    if not keywords_path.exists():
        raise FileNotFoundError(f"Keywords file not found: {keywords_path}")

    with keywords_path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel_tab

        reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError(f"Could not read headers from keywords file: {keywords_path}")

        normalized_headers = {
            header.strip().lower(): header
            for header in reader.fieldnames
            if header is not None
        }

        image_id_column = _find_header(
            normalized_headers,
            ("photo_id", "image_id", "photoid", "id"),
        )
        keyword_column = _find_header(
            normalized_headers,
            ("keyword", "keywords", "term", "tag"),
        )

        if image_id_column is None or keyword_column is None:
            raise ValueError(
                "Could not detect photo-id and keyword columns in keywords file"
            )

        keywords_by_image_id: dict[str, set[str]] = {}

        for row in reader:
            raw_image_id = (row.get(image_id_column) or "").strip()
            raw_keyword = (row.get(keyword_column) or "").strip().lower()

            if not raw_image_id or not raw_keyword:
                continue

            keywords_by_image_id.setdefault(raw_image_id, set()).add(raw_keyword)

    return {
        image_id: sorted(keywords)
        for image_id, keywords in keywords_by_image_id.items()
    }


def _find_header(
    normalized_headers: dict[str, str],
    candidates: tuple[str, ...],
) -> str | None:
    for candidate in candidates:
        if candidate in normalized_headers:
            return normalized_headers[candidate]
    return None


class QdrantStore:
    def __init__(
        self,
        collection_name: str = "photos",
        qdrant_path: Path | None = None,
        qdrant_url: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.qdrant_path = Path(qdrant_path) if qdrant_path is not None else None
        self.qdrant_url = qdrant_url
        self.client = self._create_client()

    def recreate_collection(self, vector_size: int = 512) -> None:
        if vector_size <= 0:
            raise ValueError(f"vector_size must be greater than 0, got {vector_size}")

        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(
            "Qdrant collection recreated: %s vector_size=%s",
            self.collection_name,
            vector_size,
        )

    def upload_points(
        self,
        embeddings: np.ndarray,
        image_ids: np.ndarray | list[str],
        payloads: list[dict[str, Any]],
        batch_size: int = 128,
    ) -> None:
        self.validate_collection_exists()

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings array is empty")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")

        image_id_list = [str(image_id) for image_id in np.asarray(image_ids).tolist()]

        if embeddings.shape[0] != len(image_id_list):
            raise ValueError(
                "Embeddings and image ID counts do not match: "
                f"{embeddings.shape[0]} != {len(image_id_list)}"
            )
        if embeddings.shape[0] != len(payloads):
            raise ValueError(
                "Embeddings and payload counts do not match: "
                f"{embeddings.shape[0]} != {len(payloads)}"
            )

        vector_matrix = np.ascontiguousarray(embeddings.astype(np.float32, copy=False))

        for start in range(0, vector_matrix.shape[0], batch_size):
            end = min(start + batch_size, vector_matrix.shape[0])
            batch_points = [
                models.PointStruct(
                    id=index,
                    vector=vector_matrix[index].tolist(),
                    payload=payloads[index],
                )
                for index in range(start, end)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch_points,
                wait=True,
            )

        logger.info(
            "Uploaded %s points into Qdrant collection %s",
            vector_matrix.shape[0],
            self.collection_name,
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[models.ScoredPoint]:
        self.validate_collection_exists()

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")

        vector = self._prepare_query_vector(query_vector)
        query_filter = self._build_filter(filters)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)

    def count(self) -> int:
        self.validate_collection_exists()
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)

    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection_name))

    def validate_collection_exists(self) -> None:
        if not self.collection_exists():
            raise RuntimeError(
                f"Qdrant collection does not exist: {self.collection_name}. "
                "Run scripts/setup_qdrant_collection.py first."
            )

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            logger.debug("Ignoring Qdrant close failure during shutdown", exc_info=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _create_client(self):
        if self.qdrant_url:
            return create_qdrant_client(
                QdrantSettings(
                    mode="server",
                    collection_name=self.collection_name,
                    path=self.qdrant_path or Path("data/qdrant"),
                    url=self.qdrant_url,
                )
            )
        if self.qdrant_path is None:
            from qdrant_client import QdrantClient

            return QdrantClient(":memory:")

        return create_qdrant_client(
            QdrantSettings(
                mode="local",
                collection_name=self.collection_name,
                path=self.qdrant_path,
                url="",
            )
        )

    @staticmethod
    def _prepare_query_vector(query_vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.ndim != 1:
            raise ValueError(
                "query_vector must be a 1D vector or single-row 2D array, "
                f"got shape {vector.shape}"
            )
        if vector.shape[0] == 0:
            raise ValueError("query_vector must not be empty")
        return np.ascontiguousarray(vector)

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
        if not filters:
            return None

        conditions: list[models.FieldCondition] = []

        keyword_filter = filters.get("keyword_filter")
        if keyword_filter:
            conditions.append(
                models.FieldCondition(
                    key="keywords",
                    match=models.MatchAny(any=[str(keyword_filter).lower()]),
                )
            )

        object_filter = filters.get("object_filter")
        if object_filter:
            conditions.append(
                models.FieldCondition(
                    key="detected_objects",
                    match=models.MatchAny(any=[str(object_filter).lower()]),
                )
            )

        for style_field in STYLE_FIELDS:
            min_key = f"min_{style_field}"
            max_key = f"max_{style_field}"
            min_value = filters.get(min_key)
            max_value = filters.get(max_key)

            if min_value is None and max_value is None:
                continue

            conditions.append(
                models.FieldCondition(
                    key=style_field,
                    range=models.Range(
                        gte=float(min_value) if min_value is not None else None,
                        lte=float(max_value) if max_value is not None else None,
                    ),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)
