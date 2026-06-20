from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

from app.ml.clip_encoder import CLIPEncoder
from app.ml.visual_descriptor import VisualDescriptorExtractor
from app.search.metadata_repository import ImageMetadata
from app.search.qdrant_store import QdrantStore
from app.search.style_reranker import RerankCandidate, StyleReranker


logger = logging.getLogger(__name__)


class QdrantRetrievalService:
    def __init__(
        self,
        clip_encoder: CLIPEncoder,
        qdrant_store: QdrantStore,
        style_reranker: StyleReranker | None = None,
        rerank_enabled: bool = True,
        candidate_pool_size: int = 100,
        descriptor_extractor: VisualDescriptorExtractor | None = None,
    ) -> None:
        self.clip_encoder = clip_encoder
        self.qdrant_store = qdrant_store
        self.style_reranker = style_reranker or StyleReranker()
        self.rerank_enabled = rerank_enabled
        self.candidate_pool_size = candidate_pool_size
        self.descriptor_extractor = descriptor_extractor or VisualDescriptorExtractor()

        if self.candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be greater than 0")

    def search_by_text(
        self,
        text: str,
        top_k: int = 10,
        keyword_filter: str | None = None,
        object_filter: str | None = None,
        rerank: bool = False,
        candidate_pool_size: int = 100,
        min_brightness: float | None = None,
        max_brightness: float | None = None,
        min_contrast: float | None = None,
        max_contrast: float | None = None,
        min_saturation: float | None = None,
        max_saturation: float | None = None,
        min_warmth: float | None = None,
        max_warmth: float | None = None,
    ) -> list[dict[str, Any]]:
        if not text or not text.strip():
            raise ValueError("Text query must not be empty")
        self._validate_top_k(top_k)

        query_embedding = self.clip_encoder.encode_text(text.strip())
        query_metadata = self._build_text_query_metadata(text)

        return self._search(
            query_embedding=query_embedding,
            top_k=top_k,
            query_metadata=query_metadata,
            rerank=rerank,
            candidate_pool_size=candidate_pool_size,
            keyword_filter=keyword_filter,
            object_filter=object_filter,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            min_contrast=min_contrast,
            max_contrast=max_contrast,
            min_saturation=min_saturation,
            max_saturation=max_saturation,
            min_warmth=min_warmth,
            max_warmth=max_warmth,
        )

    def search_by_image(
        self,
        image_path: Path,
        top_k: int = 10,
        keyword_filter: str | None = None,
        object_filter: str | None = None,
        rerank: bool = False,
        candidate_pool_size: int = 100,
        min_brightness: float | None = None,
        max_brightness: float | None = None,
        min_contrast: float | None = None,
        max_contrast: float | None = None,
        min_saturation: float | None = None,
        max_saturation: float | None = None,
        min_warmth: float | None = None,
        max_warmth: float | None = None,
    ) -> list[dict[str, Any]]:
        query_path = Path(image_path)
        if not query_path.exists():
            raise FileNotFoundError(f"Query image not found: {query_path}")
        self._validate_top_k(top_k)

        with Image.open(query_path) as image:
            rgb_image = image.convert("RGB")
            query_embedding = self.clip_encoder.encode_image(rgb_image)
            query_metadata = self._build_image_query_metadata(query_path, rgb_image)

        return self._search(
            query_embedding=query_embedding,
            top_k=top_k,
            query_metadata=query_metadata,
            rerank=rerank,
            candidate_pool_size=candidate_pool_size,
            keyword_filter=keyword_filter,
            object_filter=object_filter,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            min_contrast=min_contrast,
            max_contrast=max_contrast,
            min_saturation=min_saturation,
            max_saturation=max_saturation,
            min_warmth=min_warmth,
            max_warmth=max_warmth,
        )

    def close(self) -> None:
        self.qdrant_store.close()

    def _search(
        self,
        query_embedding,
        top_k: int,
        query_metadata: ImageMetadata | None,
        rerank: bool,
        candidate_pool_size: int,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        should_rerank = self._should_rerank(query_metadata=query_metadata, rerank=rerank)
        requested_candidates = max(top_k, candidate_pool_size, self.candidate_pool_size)
        search_top_k = requested_candidates if should_rerank else top_k

        points = self.qdrant_store.search(
            query_vector=query_embedding,
            top_k=search_top_k,
            filters=filters,
        )

        candidates = [
            RerankCandidate(
                metadata=self._payload_to_metadata(point.payload),
                semantic_score=float(point.score),
                final_score=float(point.score),
            )
            for point in points
        ]

        if should_rerank:
            assert query_metadata is not None
            candidates = self.style_reranker.rerank(query_metadata, candidates)

        return [
            self._candidate_to_result(candidate)
            for candidate in candidates[:top_k]
        ]

    def _candidate_to_result(self, candidate: RerankCandidate) -> dict[str, Any]:
        metadata = candidate.metadata
        return {
            "image_id": metadata.image_id,
            "file_path": metadata.file_path,
            "score": candidate.score,
            "semantic_score": candidate.semantic_score,
            "style_score": candidate.style_score,
            "final_score": candidate.final_score if candidate.final_score is not None else candidate.score,
            "keywords": list(metadata.keywords or ()),
            "detected_objects": list(metadata.detected_objects or ()),
            "brightness": metadata.brightness,
            "contrast": metadata.contrast,
            "saturation": metadata.saturation,
            "warmth": metadata.warmth,
        }

    def _payload_to_metadata(self, payload: dict[str, Any] | None) -> ImageMetadata:
        if payload is None:
            raise ValueError("Qdrant result payload is missing")

        return ImageMetadata(
            image_id=str(payload.get("image_id", "")),
            file_path=str(payload.get("file_path", "")),
            brightness=self._to_optional_float(payload.get("brightness")),
            contrast=self._to_optional_float(payload.get("contrast")),
            saturation=self._to_optional_float(payload.get("saturation")),
            warmth=self._to_optional_float(payload.get("warmth")),
            color_histogram=self._to_histogram_tuple(payload.get("color_histogram")),
            keywords=tuple(str(value) for value in (payload.get("keywords") or [])),
            detected_objects=tuple(
                str(value) for value in (payload.get("detected_objects") or [])
            ),
        )

    @staticmethod
    def _to_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _to_histogram_tuple(value: Any) -> tuple[float, ...] | None:
        if value is None:
            return None
        return tuple(float(item) for item in value)

    def _build_image_query_metadata(self, query_path: Path, image: Image.Image) -> ImageMetadata:
        descriptors = self.descriptor_extractor.extract(image)
        return ImageMetadata(
            image_id="__query_image__",
            file_path=str(query_path),
            brightness=float(descriptors["brightness"]),
            contrast=float(descriptors["contrast"]),
            saturation=float(descriptors["saturation"]),
            warmth=float(descriptors["warmth"]),
            color_histogram=tuple(float(value) for value in descriptors["color_histogram"]),
        )

    def _build_text_query_metadata(self, text: str) -> ImageMetadata | None:
        query_text = text.lower()

        brightness = None
        contrast = None
        saturation = None
        warmth = None

        if any(token in query_text for token in ("dark", "moody", "night", "low-key")):
            brightness = 0.20
            contrast = 0.70
        if any(token in query_text for token in ("bright", "sunny", "high-key")):
            brightness = 0.80
        if any(token in query_text for token in ("warm", "golden", "sunset", "tropical")):
            warmth = 0.80
        if any(token in query_text for token in ("cold", "snowy", "winter", "icy")):
            warmth = 0.20
        if any(token in query_text for token in ("vibrant", "colorful", "neon", "tropical")):
            saturation = 0.80
        if any(token in query_text for token in ("muted", "minimal", "pastel", "foggy")):
            saturation = 0.30
        if any(token in query_text for token in ("cinematic", "dramatic")) and contrast is None:
            contrast = 0.65

        if all(value is None for value in (brightness, contrast, saturation, warmth)):
            return None

        return ImageMetadata(
            image_id="__query_text__",
            file_path=text,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            warmth=warmth,
            color_histogram=None,
        )

    def _should_rerank(
        self,
        query_metadata: ImageMetadata | None,
        rerank: bool,
    ) -> bool:
        return bool((self.rerank_enabled or rerank) and rerank and query_metadata is not None)

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")
