from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RELATED_OBJECTS = {
    "person": {"human", "man", "woman", "boy", "girl"},
    "car": {"automobile", "vehicle", "truck", "bus", "van"},
    "dog": {"canine", "puppy"},
    "cat": {"kitten"},
    "bird": {"avian"},
    "building": {"architecture", "house", "tower"},
}


@dataclass(frozen=True)
class ObjectAwareScore:
    semantic_score: float
    object_score: float
    keyword_score: float
    style_score: float
    final_score: float


class ObjectAwareReranker:
    def __init__(
        self,
        semantic_weight: float = 0.55,
        object_weight: float = 0.25,
        keyword_weight: float = 0.10,
        style_weight: float = 0.10,
    ) -> None:
        total = semantic_weight + object_weight + keyword_weight + style_weight
        if total <= 0:
            raise ValueError("Object-aware reranker weights must sum to more than 0")
        self.semantic_weight = semantic_weight / total
        self.object_weight = object_weight / total
        self.keyword_weight = keyword_weight / total
        self.style_weight = style_weight / total

    @classmethod
    def object_heavy(cls) -> "ObjectAwareReranker":
        return cls(
            semantic_weight=0.45,
            object_weight=0.35,
            keyword_weight=0.10,
            style_weight=0.10,
        )

    def rerank(
        self,
        results: list[dict[str, Any]],
        *,
        requested_object: str | None = None,
        requested_keyword: str | None = None,
        style_scores_by_image_id: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        scored_results = []
        for result in results:
            image_id = str(result.get("image_id", ""))
            semantic_score = float(result.get("semantic_score", result.get("score", 0.0)) or 0.0)
            object_score = self.object_score(
                requested_object=requested_object,
                detected_objects=result.get("detected_objects") or [],
            )
            keyword_score = self.keyword_score(
                requested_keyword=requested_keyword,
                keywords=result.get("keywords") or [],
            )
            style_score = float((style_scores_by_image_id or {}).get(image_id, 0.0))
            final_score = (
                self.semantic_weight * semantic_score
                + self.object_weight * object_score
                + self.keyword_weight * keyword_score
                + self.style_weight * style_score
            )
            updated = dict(result)
            updated["object_score"] = object_score
            updated["keyword_score"] = keyword_score
            updated["object_aware_style_score"] = style_score
            updated["final_score"] = final_score
            updated["score"] = final_score
            scored_results.append(updated)

        return sorted(scored_results, key=lambda row: float(row["final_score"]), reverse=True)

    @staticmethod
    def object_score(
        *,
        requested_object: str | None,
        detected_objects: list[str] | tuple[str, ...],
    ) -> float:
        if not requested_object:
            return 0.0
        requested = requested_object.strip().lower()
        detected = {str(value).strip().lower() for value in detected_objects if str(value).strip()}
        if requested in detected:
            return 1.0
        if detected.intersection(RELATED_OBJECTS.get(requested, set())):
            return 0.5
        return 0.0

    @staticmethod
    def keyword_score(
        *,
        requested_keyword: str | None,
        keywords: list[str] | tuple[str, ...],
    ) -> float:
        if not requested_keyword:
            return 0.0
        requested = requested_keyword.strip().lower()
        normalized_keywords = {
            str(value).strip().lower()
            for value in keywords
            if str(value).strip()
        }
        return 1.0 if requested in normalized_keywords else 0.0
