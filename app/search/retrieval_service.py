from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.clip_encoder import CLIPEncoder
from app.ml.visual_descriptor import VisualDescriptorExtractor
from app.search.faiss_index import FaissIndex
from app.search.metadata_repository import ImageMetadata, MetadataRepository
from app.search.style_reranker import RerankCandidate, StyleReranker
from app.search.vector_store import VectorStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    image_id: str
    file_path: str
    score: float
    semantic_score: float
    style_score: float | None = None
    final_score: float | None = None

    def to_dict(self) -> dict[str, str | float]:
        result: dict[str, str | float] = {
            "image_id": self.image_id,
            "file_path": self.file_path,
            "score": self.score,
            "semantic_score": self.semantic_score,
            "final_score": self.final_score if self.final_score is not None else self.score,
        }
        if self.style_score is not None:
            result["style_score"] = self.style_score
        return result


class RetrievalService:
    def __init__(
        self,
        clip_encoder: CLIPEncoder,
        vector_store: VectorStore,
        faiss_index: FaissIndex,
        metadata_repository: MetadataRepository,
        style_reranker: StyleReranker | None = None,
        rerank_enabled: bool = True,
        candidate_pool_size: int = 100,
        descriptor_extractor: VisualDescriptorExtractor | None = None,
    ) -> None:
        self.clip_encoder = clip_encoder
        self.vector_store = vector_store
        self.faiss_index = faiss_index
        self.metadata_repository = metadata_repository
        self.style_reranker = style_reranker or StyleReranker()
        self.rerank_enabled = rerank_enabled
        self.candidate_pool_size = candidate_pool_size
        self.descriptor_extractor = descriptor_extractor or VisualDescriptorExtractor()

        if self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be greater than 0")

    def search_by_image(
        self,
        image_path: str | Path,
        top_k: int = 10,
        rerank_enabled: bool | None = None,
    ) -> list[dict[str, str | float]]:
        query_path = Path(image_path)
        if not query_path.exists():
            raise FileNotFoundError(f"Query image not found: {query_path}")
        self._validate_top_k(top_k)

        with Image.open(query_path) as image:
            rgb_image = image.convert("RGB")
            embedding = self.clip_encoder.encode_image(rgb_image)
            query_metadata = self._build_query_metadata(query_path, rgb_image)

        return self._search(
            query_embedding=embedding,
            top_k=top_k,
            query_metadata=query_metadata,
            rerank_enabled=rerank_enabled,
        )

    def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        rerank_enabled: bool | None = None,
    ) -> list[dict[str, str | float]]:
        if not text or not text.strip():
            raise ValueError("Text query must not be empty")
        self._validate_top_k(top_k)

        embedding = self.clip_encoder.encode_text(text.strip())
        return self._search(
            query_embedding=embedding,
            top_k=top_k,
            query_metadata=None,
            rerank_enabled=rerank_enabled,
        )

    def _search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        query_metadata: ImageMetadata | None,
        rerank_enabled: bool | None,
    ) -> list[dict[str, str | float]]:
        query_vector = np.ascontiguousarray(query_embedding.astype(np.float32, copy=False))
        image_ids = self.vector_store.get_image_ids()
        should_rerank = self._should_rerank(query_metadata, rerank_enabled)
        search_top_k = min(
            len(image_ids),
            max(top_k, self.candidate_pool_size) if should_rerank else top_k,
        )
        distances, indices = self.faiss_index.search(query_vector, top_k=search_top_k)

        ranked_ids: list[str] = []
        ranked_scores: list[float] = []

        for score, index in zip(distances, indices, strict=True):
            if index < 0:
                continue
            if index >= len(image_ids):
                logger.warning("Skipping out-of-range FAISS result index: %s", index)
                continue

            ranked_ids.append(str(image_ids[index]))
            ranked_scores.append(float(score))

        metadata_by_id = self.metadata_repository.get_many(ranked_ids)
        candidates: list[RerankCandidate] = []

        for image_id, score in zip(ranked_ids, ranked_scores, strict=True):
            metadata = metadata_by_id.get(image_id)
            if metadata is None:
                logger.warning("Metadata missing for image_id=%s", image_id)
                continue
            candidates.append(
                RerankCandidate(
                    metadata=metadata,
                    semantic_score=score,
                    final_score=score,
                )
            )

        if should_rerank:
            assert query_metadata is not None
            candidates = self.style_reranker.rerank(query_metadata, candidates)

        results = [
            SearchResult(
                image_id=candidate.metadata.image_id,
                file_path=candidate.metadata.file_path,
                score=candidate.score,
                semantic_score=candidate.semantic_score,
                style_score=candidate.style_score,
                final_score=candidate.final_score,
            )
            for candidate in candidates[:top_k]
        ]

        logger.info("Search completed: requested_top_k=%s returned=%s", top_k, len(results))
        return [result.to_dict() for result in results]

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")

    def _should_rerank(
        self,
        query_metadata: ImageMetadata | None,
        rerank_enabled: bool | None,
    ) -> bool:
        enabled = self.rerank_enabled if rerank_enabled is None else rerank_enabled
        return bool(enabled and query_metadata is not None and query_metadata.has_style_features)

    def _build_query_metadata(self, query_path: Path, image: Image.Image) -> ImageMetadata:
        descriptors = self.descriptor_extractor.extract(image)
        return ImageMetadata(
            image_id="__query__",
            file_path=str(query_path),
            brightness=float(descriptors["brightness"]),
            contrast=float(descriptors["contrast"]),
            saturation=float(descriptors["saturation"]),
            warmth=float(descriptors["warmth"]),
            color_histogram=tuple(float(value) for value in descriptors["color_histogram"]),
        )
