from __future__ import annotations

import math

import pandas as pd
import pytest

from experiments.scripts.evaluate_search_quality import (
    aggregate_by_system,
    dcg_at_k,
    evaluate_all_queries,
    join_results_with_labels,
    mean_relevance_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    success_at_k,
    validate_joined_labels,
    validate_result_ranks,
)


def test_precision_success_mrr_and_mean_relevance() -> None:
    relevances = [2, 1, 0, 2, 0]

    assert precision_at_k(relevances, 5, relevance_threshold=1) == 3 / 5
    assert precision_at_k(relevances, 10, relevance_threshold=1) == 3 / 5
    assert precision_at_k(relevances, 5, relevance_threshold=2) == 2 / 5
    assert reciprocal_rank(relevances, relevance_threshold=1) == 1.0
    assert reciprocal_rank([0, 0, 2], relevance_threshold=1) == 1 / 3
    assert success_at_k(relevances, 1, relevance_threshold=1) == 1.0
    assert success_at_k([0, 0, 2], 2, relevance_threshold=1) == 0.0
    assert mean_relevance_at_k(relevances, 10) == 1.0


def test_no_relevant_results_and_fewer_than_k_results() -> None:
    relevances = [0, 0]

    assert precision_at_k(relevances, 5, relevance_threshold=1) == 0.0
    assert reciprocal_rank(relevances, relevance_threshold=1) == 0.0
    assert success_at_k(relevances, 5, relevance_threshold=1) == 0.0
    assert ndcg_at_k(relevances, 5) == 0.0
    assert mean_relevance_at_k(relevances, 10) == 0.0


def test_ndcg_uses_graded_relevance_ordering() -> None:
    relevances = [2, 1, 0, 2, 0]
    expected_dcg = (
        (2**2 - 1) / math.log2(2)
        + (2**1 - 1) / math.log2(3)
        + (2**0 - 1) / math.log2(4)
        + (2**2 - 1) / math.log2(5)
        + (2**0 - 1) / math.log2(6)
    )
    expected_idcg = dcg_at_k([2, 2, 1, 0, 0], 5)

    assert dcg_at_k(relevances, 5) == pytest.approx(expected_dcg)
    assert ndcg_at_k(relevances, 5) == pytest.approx(expected_dcg / expected_idcg)


def test_duplicate_result_and_rank_problem_detection() -> None:
    results = pd.DataFrame(
        [
            {"query_id": "Q1", "system": "sys", "rank": "1", "image_id": "a"},
            {"query_id": "Q1", "system": "sys", "rank": "1", "image_id": "a"},
            {"query_id": "Q1", "system": "sys", "rank": "3", "image_id": "b"},
        ]
    )
    results["rank_number"] = pd.to_numeric(results["rank"], errors="coerce")

    quality = validate_result_ranks(results)

    assert len(quality["duplicate_system_query_image_rows"]) == 2
    assert len(quality["duplicate_ranks"]) == 1
    assert len(quality["rank_gaps"]) == 1
    assert quality["rank_gaps"].iloc[0]["missing_ranks"] == "2"


def test_incomplete_label_handling() -> None:
    results = pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "query": "dog",
                "type": "object_text",
                "system": "sys",
                "rank": "1",
                "image_id": "a",
                "score": "0.1",
                "rank_number": 1,
            }
        ]
    )
    judgments = pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "image_id": "a",
                "relevance": "",
                "confidence": "",
                "comment": "",
                "query": "dog",
                "type": "object_text",
            }
        ]
    )

    joined, _ = join_results_with_labels(results, judgments)

    with pytest.raises(ValueError):
        validate_joined_labels(joined, allow_incomplete_labels=False)
    validate_joined_labels(joined, allow_incomplete_labels=True)


def test_macro_averaging_across_queries() -> None:
    metrics_by_query = pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "query": "a",
                "type": "semantic_text",
                "system": "sys",
                "result_count": 2,
                "labeled_result_count": 2,
                "precision_at_5": 1.0,
                "precision_at_10": 1.0,
                "ndcg_at_5": 1.0,
                "ndcg_at_10": 1.0,
                "mrr": 1.0,
                "success_at_1": 1.0,
                "success_at_5": 1.0,
                "mean_relevance_at_10": 2.0,
            },
            {
                "query_id": "Q2",
                "query": "b",
                "type": "semantic_text",
                "system": "sys",
                "result_count": 10,
                "labeled_result_count": 10,
                "precision_at_5": 0.0,
                "precision_at_10": 0.0,
                "ndcg_at_5": 0.0,
                "ndcg_at_10": 0.0,
                "mrr": 0.0,
                "success_at_1": 0.0,
                "success_at_5": 0.0,
                "mean_relevance_at_10": 0.0,
            },
        ]
    )

    by_system = aggregate_by_system(metrics_by_query)

    assert by_system.iloc[0]["query_count"] == 2
    assert by_system.iloc[0]["precision_at_10"] == 0.5
    assert by_system.iloc[0]["mean_relevance_at_10"] == 1.0


def test_evaluate_query_system_filters_unlabeled_rows_when_allowed() -> None:
    joined = pd.DataFrame(
        [
            {
                "query_id": "Q1",
                "query": "dog",
                "type": "object_text",
                "system": "sys",
                "rank": "1",
                "rank_number": 1,
                "image_id": "a",
                "relevance_text": "",
                "relevance_value": float("nan"),
            },
            {
                "query_id": "Q1",
                "query": "dog",
                "type": "object_text",
                "system": "sys",
                "rank": "2",
                "rank_number": 2,
                "image_id": "b",
                "relevance_text": "2",
                "relevance_value": 2,
            },
        ]
    )

    metrics = evaluate_all_queries(joined, relevance_threshold=1, allow_incomplete_labels=True)

    assert metrics.iloc[0]["result_count"] == 2
    assert metrics.iloc[0]["labeled_result_count"] == 1
    assert metrics.iloc[0]["precision_at_5"] == 1.0
    assert metrics.iloc[0]["ndcg_at_5"] == 1.0
