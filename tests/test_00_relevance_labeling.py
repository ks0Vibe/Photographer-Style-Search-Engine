from __future__ import annotations

import pandas as pd

from experiments.scripts.check_relevance_labeling import validate_labels
from experiments.scripts.prepare_relevance_labeling import (
    JUDGMENT_COLUMNS,
    aggregate_judgments,
    normalize_validation_results,
    preserve_previous_labels,
)


def test_duplicate_query_image_rows_are_merged_and_labels_preserved() -> None:
    raw_results = normalize_validation_results(
        pd.DataFrame(
            [
                {
                    "query_id": "Q001",
                    "query": "dog on beach",
                    "type": "object_text",
                    "system": "faiss_baseline",
                    "rank": "3",
                    "image_id": "img/a",
                    "score": "0.2",
                    "image_path": "a.jpg",
                    "matched_keywords": "dog;beach",
                    "detected_objects": "dog",
                    "brightness": "0.8",
                    "contrast": "0.2",
                    "saturation": "0.3",
                    "warmth": "0.4",
                },
                {
                    "query_id": "Q001",
                    "query": "dog on beach",
                    "type": "object_text",
                    "system": "qdrant_semantic",
                    "rank": "1",
                    "image_id": "img/a",
                    "score": "0.5",
                    "image_path": "a.jpg",
                    "matched_keywords": "sand",
                    "detected_objects": "dog",
                    "brightness": "0.8",
                    "contrast": "0.2",
                    "saturation": "0.3",
                    "warmth": "0.4",
                },
                {
                    "query_id": "Q001",
                    "query": "dog on beach",
                    "type": "object_text",
                    "system": "qdrant_semantic",
                    "rank": "2",
                    "image_id": "img-b",
                    "score": "0.4",
                    "image_path": "b.jpg",
                    "matched_keywords": "",
                    "detected_objects": "",
                    "brightness": "",
                    "contrast": "",
                    "saturation": "",
                    "warmth": "",
                },
            ]
        )
    )

    judgments = aggregate_judgments(raw_results, max_items_per_query=50)
    judgments = preserve_previous_labels(
        judgments,
        {
            "q001__img_a": {
                "relevance": "2",
                "confidence": "high",
                "comment": "clear dog on beach",
            }
        },
    )

    assert len(raw_results) == 3
    assert len(judgments) == 2

    merged = judgments[judgments["judgment_id"] == "q001__img_a"].iloc[0]
    assert merged["source_systems"] == "qdrant_semantic|faiss_baseline"
    assert merged["best_rank"] == "1"
    assert merged["ranks_by_system"] == "faiss_baseline:3|qdrant_semantic:1"
    assert merged["max_score"] == "0.500000"
    assert merged["matched_keywords"] == "sand;dog;beach"
    assert merged["relevance"] == "2"
    assert merged["confidence"] == "high"
    assert merged["comment"] == "clear dog on beach"


def test_missing_optional_metadata_columns_do_not_crash() -> None:
    raw_results = normalize_validation_results(
        pd.DataFrame(
            [
                {
                    "query_id": "Q002",
                    "query": "warm portrait",
                    "type": "semantic_text",
                    "system": "faiss_baseline",
                    "rank": "1",
                    "image_id": "img-1",
                }
            ]
        )
    )
    judgments = aggregate_judgments(raw_results, max_items_per_query=50)

    assert len(judgments) == 1
    assert judgments.loc[0, "image_path"] == ""
    assert judgments.loc[0, "matched_keywords"] == ""
    assert judgments.loc[0, "detected_objects"] == ""


def test_invalid_relevance_and_confidence_values_are_detected() -> None:
    labels = pd.DataFrame(
        [
            {
                **{column: "" for column in JUDGMENT_COLUMNS},
                "judgment_id": "q001__img-1",
                "query_id": "Q001",
                "type": "semantic_text",
                "image_id": "img-1",
                "relevance": "3",
                "confidence": "certain",
            }
        ],
        columns=JUDGMENT_COLUMNS,
    )

    summary = validate_labels(labels)

    assert summary["invalid_rows"]
    assert summary["has_fatal_label_values"] is True
    assert any("invalid relevance values" in item for item in summary["invalid_rows"])
    assert any("invalid confidence values" in item for item in summary["invalid_rows"])
