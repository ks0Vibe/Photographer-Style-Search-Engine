from __future__ import annotations

from dataclasses import dataclass

from app.search.metadata_repository import ImageMetadata
from app.search.style_similarity import StyleSimilarity


@dataclass(frozen=True)
class RerankCandidate:
    metadata: ImageMetadata
    semantic_score: float
    style_score: float | None = None
    final_score: float | None = None

    @property
    def score(self) -> float:
        return self.final_score if self.final_score is not None else self.semantic_score


class StyleReranker:
    def __init__(
        self,
        style_similarity: StyleSimilarity | None = None,
        semantic_weight: float = 0.75,
        style_weight: float = 0.25,
    ) -> None:
        total_weight = semantic_weight + style_weight
        if total_weight <= 0:
            raise ValueError("semantic_weight + style_weight must be greater than 0")

        self.style_similarity = style_similarity or StyleSimilarity()
        self.semantic_weight = semantic_weight / total_weight
        self.style_weight = style_weight / total_weight

    def rerank(
        self,
        query_metadata: ImageMetadata,
        candidates: list[RerankCandidate],
    ) -> list[RerankCandidate]:
        reranked_candidates: list[RerankCandidate] = []

        for candidate in candidates:
            style_score = self.style_similarity.compute_style_score(
                query_metadata=query_metadata,
                candidate_metadata=candidate.metadata,
            )
            final_score = (
                self.semantic_weight * candidate.semantic_score
                + self.style_weight * style_score
            )

            reranked_candidates.append(
                RerankCandidate(
                    metadata=candidate.metadata,
                    semantic_score=candidate.semantic_score,
                    style_score=style_score,
                    final_score=final_score,
                )
            )

        return sorted(reranked_candidates, key=lambda candidate: candidate.score, reverse=True)
