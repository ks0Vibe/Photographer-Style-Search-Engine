from __future__ import annotations

import math

import numpy as np

from app.search.metadata_repository import ImageMetadata


class StyleSimilarity:
    def brightness_similarity(self, a: float, b: float) -> float:
        return self._bounded_difference_similarity(a, b)

    def contrast_similarity(self, a: float, b: float) -> float:
        return self._bounded_difference_similarity(a, b)

    def saturation_similarity(self, a: float, b: float) -> float:
        return self._bounded_difference_similarity(a, b)

    def warmth_similarity(self, a: float, b: float) -> float:
        return self._bounded_difference_similarity(a, b)

    def histogram_similarity(
        self,
        histogram_a: tuple[float, ...],
        histogram_b: tuple[float, ...],
    ) -> float:
        vector_a = np.asarray(histogram_a, dtype=np.float32)
        vector_b = np.asarray(histogram_b, dtype=np.float32)

        if vector_a.ndim != 1 or vector_b.ndim != 1:
            raise ValueError("Color histograms must be 1D sequences")
        if vector_a.shape != vector_b.shape:
            raise ValueError(
                f"Color histogram shape mismatch: {vector_a.shape} != {vector_b.shape}"
            )

        norm_a = float(np.linalg.norm(vector_a))
        norm_b = float(np.linalg.norm(vector_b))
        if math.isclose(norm_a, 0.0) or math.isclose(norm_b, 0.0):
            return 0.0

        cosine_similarity = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
        return self._clamp(cosine_similarity)

    def compute_style_score(
        self,
        query_metadata: ImageMetadata,
        candidate_metadata: ImageMetadata,
    ) -> float:
        if not query_metadata.has_style_features or not candidate_metadata.has_style_features:
            return 0.0

        assert query_metadata.brightness is not None
        assert query_metadata.contrast is not None
        assert query_metadata.saturation is not None
        assert query_metadata.warmth is not None
        assert query_metadata.color_histogram is not None
        assert candidate_metadata.brightness is not None
        assert candidate_metadata.contrast is not None
        assert candidate_metadata.saturation is not None
        assert candidate_metadata.warmth is not None
        assert candidate_metadata.color_histogram is not None

        scores = (
            self.brightness_similarity(query_metadata.brightness, candidate_metadata.brightness),
            self.contrast_similarity(query_metadata.contrast, candidate_metadata.contrast),
            self.saturation_similarity(query_metadata.saturation, candidate_metadata.saturation),
            self.warmth_similarity(query_metadata.warmth, candidate_metadata.warmth),
            self.histogram_similarity(
                query_metadata.color_histogram,
                candidate_metadata.color_histogram,
            ),
        )

        return float(sum(scores) / len(scores))

    @staticmethod
    def _bounded_difference_similarity(a: float, b: float) -> float:
        return StyleSimilarity._clamp(1.0 - abs(a - b))

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
