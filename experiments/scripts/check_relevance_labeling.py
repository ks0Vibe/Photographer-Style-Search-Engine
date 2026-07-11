from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS_PATH = PROJECT_ROOT / "experiments" / "10_relevance_labeling" / "relevance_judgments.csv"

REQUIRED_COLUMNS = [
    "judgment_id",
    "query_id",
    "query",
    "type",
    "image_id",
    "image_path",
    "source_systems",
    "best_rank",
    "ranks_by_system",
    "max_score",
    "matched_keywords",
    "detected_objects",
    "brightness",
    "contrast",
    "saturation",
    "warmth",
    "relevance",
    "confidence",
    "comment",
]
ALLOWED_RELEVANCE = {"", "0", "1", "2"}
ALLOWED_CONFIDENCE = {"", "high", "medium", "low"}
KNOWN_QUERY_TYPES = {"semantic_text", "style_text", "object_text", "mixed_text", "image_to_image"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate manual relevance-labeling CSV structure and labels.")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_labels(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise FileNotFoundError(f"Relevance judgments CSV not found: {labels_path}")
    return pd.read_csv(labels_path, dtype=str, keep_default_na=False)


def validate_labels(labels: pd.DataFrame) -> dict[str, Any]:
    invalid_rows: list[str] = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in labels.columns]
    if missing_columns:
        invalid_rows.append(f"missing required columns: {missing_columns}")

    working = labels.copy()
    for column in REQUIRED_COLUMNS:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].fillna("").astype(str).str.strip()

    duplicate_ids = working[working["judgment_id"].duplicated(keep=False)]["judgment_id"].unique().tolist()
    if duplicate_ids:
        invalid_rows.append(f"duplicate judgment_id values: {duplicate_ids[:10]}")

    duplicate_pairs = working[working.duplicated(subset=["query_id", "image_id"], keep=False)]
    if not duplicate_pairs.empty:
        examples = (
            duplicate_pairs[["query_id", "image_id"]]
            .drop_duplicates()
            .head(10)
            .apply(lambda row: f"{row['query_id']}::{row['image_id']}", axis=1)
            .tolist()
        )
        invalid_rows.append(f"duplicated query-image labels: {examples}")

    invalid_relevance = working[~working["relevance"].isin(ALLOWED_RELEVANCE)]
    if not invalid_relevance.empty:
        examples = invalid_relevance[["judgment_id", "relevance"]].head(10).to_dict("records")
        invalid_rows.append(f"invalid relevance values: {examples}")

    invalid_confidence = working[~working["confidence"].isin(ALLOWED_CONFIDENCE)]
    if not invalid_confidence.empty:
        examples = invalid_confidence[["judgment_id", "confidence"]].head(10).to_dict("records")
        invalid_rows.append(f"invalid confidence values: {examples}")

    empty_image_ids = working[working["image_id"] == ""]
    if not empty_image_ids.empty:
        invalid_rows.append(f"rows with empty image_id: {empty_image_ids.index[:10].tolist()}")

    empty_query_ids = working[working["query_id"] == ""]
    if not empty_query_ids.empty:
        invalid_rows.append(f"rows with empty query_id: {empty_query_ids.index[:10].tolist()}")

    unknown_types = working[~working["type"].isin(KNOWN_QUERY_TYPES)]
    if not unknown_types.empty:
        examples = unknown_types[["judgment_id", "type"]].head(10).to_dict("records")
        invalid_rows.append(f"unknown query types: {examples}")

    labeled_mask = working["relevance"] != ""
    total = len(working)
    labeled = int(labeled_mask.sum())
    unlabeled = total - labeled

    fully_labeled = 0
    partially_labeled = 0
    not_started = 0
    if total:
        for _, group in working.assign(_labeled=labeled_mask).groupby("query_id"):
            labeled_count = int(group["_labeled"].sum())
            if labeled_count == 0:
                not_started += 1
            elif labeled_count == len(group):
                fully_labeled += 1
            else:
                partially_labeled += 1

    query_count = int(working["query_id"].nunique()) if total else 0
    if total > 0 and query_count == 0:
        invalid_rows.append("no query has at least one judgment")

    return {
        "total_judgments": total,
        "labeled_judgments": labeled,
        "unlabeled_judgments": unlabeled,
        "completion_percentage": (labeled / total * 100.0) if total else 0.0,
        "queries_fully_labeled": fully_labeled,
        "queries_partially_labeled": partially_labeled,
        "queries_not_started": not_started,
        "invalid_rows": invalid_rows,
        "has_fatal_label_values": not invalid_relevance.empty or not invalid_confidence.empty,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"total judgments: {summary['total_judgments']}")
    print(f"labeled judgments: {summary['labeled_judgments']}")
    print(f"unlabeled judgments: {summary['unlabeled_judgments']}")
    print(f"completion percentage: {summary['completion_percentage']:.1f}%")
    print(f"queries fully labeled: {summary['queries_fully_labeled']}")
    print(f"queries partially labeled: {summary['queries_partially_labeled']}")
    print(f"queries not started: {summary['queries_not_started']}")
    print(f"invalid rows: {len(summary['invalid_rows'])}")
    for item in summary["invalid_rows"]:
        print(f"- {item}")


def main() -> None:
    args = parse_args()
    labels_path = resolve_project_path(args.labels_path)
    labels = load_labels(labels_path)
    summary = validate_labels(labels)
    print_summary(summary)

    if summary["invalid_rows"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
