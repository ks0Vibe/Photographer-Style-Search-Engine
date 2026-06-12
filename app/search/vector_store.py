from __future__ import annotations

import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        embeddings_path: Path,
        image_ids_path: Path,
        expected_dim: int = 512,
    ) -> None:
        self.embeddings_path = Path(embeddings_path)
        self.image_ids_path = Path(image_ids_path)
        self.expected_dim = expected_dim
        self._embeddings: np.ndarray | None = None
        self._image_ids: np.ndarray | None = None

    def load_embeddings(self) -> None:
        if self._embeddings is not None:
            return

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")

        try:
            embeddings = np.load(self.embeddings_path, allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embeddings from {self.embeddings_path}"
            ) from exc

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array, got shape {embeddings.shape}"
            )
        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings array is empty")
        if embeddings.shape[1] != self.expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dim}, "
                f"got {embeddings.shape[1]}"
            )

        embeddings = np.ascontiguousarray(embeddings.astype(np.float32, copy=False))

        self._embeddings = embeddings
        logger.info("Loaded embeddings: count=%s dim=%s", embeddings.shape[0], embeddings.shape[1])
        self._validate_alignment()

    def load_image_ids(self) -> None:
        if self._image_ids is not None:
            return

        if not self.image_ids_path.exists():
            raise FileNotFoundError(f"Image IDs file not found: {self.image_ids_path}")

        try:
            image_ids = np.load(self.image_ids_path, allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image IDs from {self.image_ids_path}"
            ) from exc

        if image_ids.ndim != 1:
            raise ValueError(f"Image IDs must be a 1D array, got shape {image_ids.shape}")
        if image_ids.shape[0] == 0:
            raise ValueError("Image IDs array is empty")

        self._image_ids = np.asarray(image_ids)
        logger.info("Loaded image IDs: count=%s", self._image_ids.shape[0])
        self._validate_alignment()

    def get_embeddings(self) -> np.ndarray:
        self.load_embeddings()
        assert self._embeddings is not None
        return self._embeddings

    def get_image_ids(self) -> np.ndarray:
        self.load_image_ids()
        assert self._image_ids is not None
        return self._image_ids

    def _validate_alignment(self) -> None:
        if self._embeddings is None or self._image_ids is None:
            return

        if self._embeddings.shape[0] != self._image_ids.shape[0]:
            raise ValueError(
                "Embedding and image ID counts do not match: "
                f"{self._embeddings.shape[0]} != {self._image_ids.shape[0]}"
            )
