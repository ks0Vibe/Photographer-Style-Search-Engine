from __future__ import annotations

import math

import numpy as np

from app.search.metadata_repository import ImageMetadata


class StyleSimilarity:
    def __init__(
        self,
        brightness_weight: float = 0.20,
        contrast_weight: float = 0.20,
        saturation_weight: float = 0.20,
        warmth_weight: float = 0.20,
        histogram_weight: float = 0.20,
    ) -> None:
        self._weights = {
            "brightness": brightness_weight,
            "contrast": contrast_weight,
            "saturation": saturation_weight,
            "warmth": warmth_weight,
            "histogram": histogram_weight,
        }
        total_weight = sum(self._weights.values())
        if total_weight <= 0:
            raise ValueError("Style similarity weights must sum to more than 0")
        self._weights = {
            key: value / total_weight
            for key, value in self._weights.items()
        }

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
        weighted_scores: list[tuple[str, float]] = []

        if query_metadata.brightness is not None and candidate_metadata.brightness is not None:
            weighted_scores.append(
                (
                    "brightness",
                    self.brightness_similarity(
                        query_metadata.brightness,
                        candidate_metadata.brightness,
                    ),
                )
            )
        if query_metadata.contrast is not None and candidate_metadata.contrast is not None:
            weighted_scores.append(
                (
                    "contrast",
                    self.contrast_similarity(
                        query_metadata.contrast,
                        candidate_metadata.contrast,
                    ),
                )
            )
        if query_metadata.saturation is not None and candidate_metadata.saturation is not None:
            weighted_scores.append(
                (
                    "saturation",
                    self.saturation_similarity(
                        query_metadata.saturation,
                        candidate_metadata.saturation,
                    ),
                )
            )
        if query_metadata.warmth is not None and candidate_metadata.warmth is not None:
            weighted_scores.append(
                (
                    "warmth",
                    self.warmth_similarity(
                        query_metadata.warmth,
                        candidate_metadata.warmth,
                    ),
                )
            )
        if (
            query_metadata.color_histogram is not None
            and candidate_metadata.color_histogram is not None
        ):
            weighted_scores.append(
                (
                    "histogram",
                    self.histogram_similarity(
                        query_metadata.color_histogram,
                        candidate_metadata.color_histogram,
                    ),
                )
            )

        if not weighted_scores:
            return 0.0

        total_weight = sum(self._weights[key] for key, _ in weighted_scores)
        if total_weight <= 0:
            return 0.0

        return float(
            sum(self._weights[key] * score for key, score in weighted_scores) / total_weight
        )

    @staticmethod
    def _bounded_difference_similarity(a: float, b: float) -> float:
        return StyleSimilarity._clamp(1.0 - abs(a - b))

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
