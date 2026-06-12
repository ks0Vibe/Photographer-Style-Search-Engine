from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np


logger = logging.getLogger(__name__)


class FaissIndex:
    def __init__(
        self,
        index_path: Path,
        dimension: int = 512,
        default_top_k: int = 10,
    ) -> None:
        self.index_path = Path(index_path)
        self.dimension = dimension
        self.default_top_k = default_top_k
        self._index: faiss.Index | None = None

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be a 2D array, got shape {vectors.shape}")
        if vectors.shape[0] == 0:
            raise ValueError("Cannot build a FAISS index from an empty vector array")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"
            )

        vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        index = self._create_index()
        index.add(vectors)
        self._index = index
        logger.info("Built FAISS Flat index with %s vectors", index.ntotal)

    def save(self) -> None:
        index = self._require_index()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        logger.info("Saved FAISS index to %s", self.index_path)

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {self.index_path}")

        try:
            index = faiss.read_index(str(self.index_path))
        except Exception as exc:
            raise RuntimeError(f"Failed to load FAISS index from {self.index_path}") from exc

        loaded_dimension = getattr(index, "d", None)
        if loaded_dimension != self.dimension:
            raise ValueError(
                f"Loaded index dimension mismatch: expected {self.dimension}, "
                f"got {loaded_dimension}"
            )

        self._index = index
        logger.info("Loaded FAISS index from %s with %s vectors", self.index_path, index.ntotal)

    def search(self, query: np.ndarray, top_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        index = self._require_index()

        if index.ntotal == 0:
            raise ValueError("Cannot search an empty FAISS index")

        search_top_k = top_k if top_k is not None else self.default_top_k
        if search_top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {search_top_k}")

        query_matrix = self._prepare_query(query)
        distances, indices = index.search(query_matrix, search_top_k)
        return distances[0], indices[0]

    def _create_index(self) -> faiss.Index:
        return faiss.IndexFlatIP(self.dimension)

    def _prepare_query(self, query: np.ndarray) -> np.ndarray:
        if query.ndim == 1:
            query_matrix = query.reshape(1, -1)
        elif query.ndim == 2 and query.shape[0] == 1:
            query_matrix = query
        else:
            raise ValueError(
                "Query must be a 1D vector or a single-row 2D array, "
                f"got shape {query.shape}"
            )

        if query_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, got {query_matrix.shape[1]}"
            )

        return np.ascontiguousarray(query_matrix.astype(np.float32, copy=False))

    def _require_index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("FAISS index is not initialized. Build or load the index first.")
        return self._index
