from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "experiments" / "09_validation_results" / "all_validation_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "10_relevance_labeling"
DEFAULT_MAX_ITEMS_PER_QUERY = 50

REQUIRED_INPUT_COLUMNS = ["query_id", "query", "type", "system", "rank", "image_id"]
OPTIONAL_INPUT_COLUMNS = [
    "score",
    "image_path",
    "matched_keywords",
    "detected_objects",
    "brightness",
    "contrast",
    "saturation",
    "warmth",
    "notes",
]
JUDGMENT_COLUMNS = [
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
MANUAL_LABEL_COLUMNS = ["relevance", "confidence", "comment"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deduplicated manual relevance-labeling pool from validation results."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items-per-query", type=int, default=DEFAULT_MAX_ITEMS_PER_QUERY)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_validation_results(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Validation results CSV not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    return normalize_validation_results(df)


def normalize_validation_results(df: pd.DataFrame) -> pd.DataFrame:
    validate_input_schema(df)
    for column in OPTIONAL_INPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    for column in REQUIRED_INPUT_COLUMNS + OPTIONAL_INPUT_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[(df["query_id"] != "") & (df["image_id"] != "")].copy()
    df["rank_number"] = pd.to_numeric(df["rank"], errors="coerce")
    df["score_number"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def validate_input_schema(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input validation results are missing required columns: {missing}")


def stable_judgment_id(query_id: str, image_id: str) -> str:
    query_part = sanitize_identifier(query_id).lower()
    image_part = sanitize_identifier(image_id)
    return f"{query_part}__{image_part}"


def sanitize_identifier(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "missing"


def first_non_empty(values: pd.Series) -> str:
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


def format_number(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def split_metadata_values(raw_value: str) -> list[str]:
    pieces = re.split(r"[;|]", str(raw_value))
    return [piece.strip() for piece in pieces if piece.strip()]


def combine_metadata_values(values: pd.Series) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        for piece in split_metadata_values(str(value)):
            key = piece.casefold()
            if key not in seen:
                seen.add(key)
                ordered.append(piece)
    return ";".join(ordered)


def aggregate_ranks(group: pd.DataFrame) -> str:
    pieces: list[str] = []
    for system in sorted(group["system"].dropna().unique()):
        system_rows = group[group["system"] == system].copy()
        numeric_ranks = pd.to_numeric(system_rows["rank"], errors="coerce").dropna()
        if not numeric_ranks.empty:
            ranks = sorted({int(rank) for rank in numeric_ranks})
            rank_text = ",".join(str(rank) for rank in ranks)
        else:
            text_ranks = sorted({str(rank).strip() for rank in system_rows["rank"] if str(rank).strip()})
            rank_text = ",".join(text_ranks)
        pieces.append(f"{system}:{rank_text}" if rank_text else f"{system}:")
    return "|".join(pieces)


def aggregate_source_systems(group: pd.DataFrame) -> str:
    system_order: list[tuple[float, str]] = []
    for system in group["system"].dropna().unique():
        system_rows = group[group["system"] == system]
        ranks = pd.to_numeric(system_rows["rank"], errors="coerce").dropna()
        best = float(ranks.min()) if not ranks.empty else float("inf")
        system_order.append((best, str(system)))
    return "|".join(system for _, system in sorted(system_order, key=lambda item: (item[0], item[1])))


def aggregate_judgments(results_df: pd.DataFrame, max_items_per_query: int) -> pd.DataFrame:
    if max_items_per_query <= 0:
        raise ValueError("--max-items-per-query must be greater than 0")

    rows: list[dict[str, str]] = []
    for (query_id, image_id), group in results_df.groupby(["query_id", "image_id"], sort=False):
        ordered_group = group.sort_values(
            by=["rank_number", "system"],
            ascending=[True, True],
            na_position="last",
            kind="stable",
        )
        best_rank_value = ordered_group["rank_number"].min()
        best_rank = "" if pd.isna(best_rank_value) else str(int(best_rank_value))
        max_score_value = ordered_group["score_number"].max()

        rows.append(
            {
                "judgment_id": stable_judgment_id(query_id, image_id),
                "query_id": str(query_id),
                "query": first_non_empty(ordered_group["query"]),
                "type": first_non_empty(ordered_group["type"]),
                "image_id": str(image_id),
                "image_path": first_non_empty(ordered_group["image_path"]),
                "source_systems": aggregate_source_systems(ordered_group),
                "best_rank": best_rank,
                "ranks_by_system": aggregate_ranks(ordered_group),
                "max_score": format_number(max_score_value),
                "matched_keywords": combine_metadata_values(ordered_group["matched_keywords"]),
                "detected_objects": combine_metadata_values(ordered_group["detected_objects"]),
                "brightness": first_non_empty(ordered_group["brightness"]),
                "contrast": first_non_empty(ordered_group["contrast"]),
                "saturation": first_non_empty(ordered_group["saturation"]),
                "warmth": first_non_empty(ordered_group["warmth"]),
                "relevance": "",
                "confidence": "",
                "comment": "",
            }
        )

    judgments = pd.DataFrame(rows, columns=JUDGMENT_COLUMNS)
    judgments = make_judgment_ids_unique(judgments)
    judgments["_best_rank_sort"] = pd.to_numeric(judgments["best_rank"], errors="coerce")
    judgments = judgments.sort_values(
        by=["query_id", "_best_rank_sort", "image_id"],
        ascending=[True, True, True],
        na_position="last",
        kind="stable",
    )
    judgments = judgments.groupby("query_id", sort=False).head(max_items_per_query).copy()
    judgments = judgments.drop(columns=["_best_rank_sort"])
    return judgments[JUDGMENT_COLUMNS].reset_index(drop=True)


def make_judgment_ids_unique(judgments: pd.DataFrame) -> pd.DataFrame:
    counts = judgments["judgment_id"].value_counts()
    duplicated_ids = set(counts[counts > 1].index)
    if not duplicated_ids:
        return judgments

    judgments = judgments.copy()
    for index, row in judgments[judgments["judgment_id"].isin(duplicated_ids)].iterrows():
        digest = hashlib.sha1(f"{row['query_id']}::{row['image_id']}".encode("utf-8")).hexdigest()[:8]
        judgments.at[index, "judgment_id"] = f"{row['judgment_id']}__{digest}"
    return judgments


def load_previous_labels(existing_path: Path) -> dict[str, dict[str, str]]:
    if not existing_path.exists():
        return {}
    previous = pd.read_csv(existing_path, dtype=str, keep_default_na=False)
    if "judgment_id" not in previous.columns:
        return {}

    for column in MANUAL_LABEL_COLUMNS:
        if column not in previous.columns:
            previous[column] = ""

    labels: dict[str, dict[str, str]] = {}
    for _, row in previous.iterrows():
        judgment_id = str(row["judgment_id"]).strip()
        if not judgment_id:
            continue
        labels[judgment_id] = {
            column: str(row.get(column, "")).strip()
            for column in MANUAL_LABEL_COLUMNS
            if str(row.get(column, "")).strip()
        }
    return labels


def preserve_previous_labels(judgments: pd.DataFrame, previous_labels: dict[str, dict[str, str]]) -> pd.DataFrame:
    judgments = judgments.copy()
    for index, row in judgments.iterrows():
        labels = previous_labels.get(str(row["judgment_id"]))
        if not labels:
            continue
        for column, value in labels.items():
            if value and not str(judgments.at[index, column]).strip():
                judgments.at[index, column] = value
    return judgments


def write_labeling_csvs(judgments: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    judgments.to_csv(output_dir / "relevance_judgments.csv", index=False)
    by_query = judgments.copy()
    by_query["_best_rank_sort"] = pd.to_numeric(by_query["best_rank"], errors="coerce")
    by_query = by_query.sort_values(
        by=["query_id", "_best_rank_sort", "source_systems"],
        ascending=[True, True, True],
        na_position="last",
        kind="stable",
    ).drop(columns=["_best_rank_sort"])
    by_query.to_csv(output_dir / "relevance_judgments_by_query.csv", index=False)

    labeled_mask = judgments["relevance"].astype(str).str.strip() != ""
    judgments[~labeled_mask].to_csv(output_dir / "unlabeled_items.csv", index=False)
    judgments[labeled_mask].to_csv(output_dir / "labeled_items.csv", index=False)


def build_progress_markdown(judgments: pd.DataFrame) -> str:
    stats = compute_progress_stats(judgments)
    timestamp = datetime.now().isoformat(timespec="seconds")

    lines = [
        "# Relevance Labeling Progress",
        "",
        f"- Generation timestamp: {timestamp}",
        f"- Number of queries: {stats['number_of_queries']}",
        f"- Total unique judgments: {stats['total_judgments']}",
        f"- Labeled judgments: {stats['labeled_judgments']}",
        f"- Unlabeled judgments: {stats['unlabeled_judgments']}",
        f"- Completion percentage: {stats['completion_percentage']:.1f}%",
        f"- Fully labeled queries: {stats['fully_labeled_queries']}",
        f"- Partially labeled queries: {stats['partially_labeled_queries']}",
        f"- Not-started queries: {stats['not_started_queries']}",
        "",
        "## Progress by Query Type",
        "",
        "| Query type | Queries | Judgments | Labeled | Completion |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in stats["query_type_rows"]:
        lines.append(
            f"| `{row['type']}` | {row['queries']} | {row['judgments']} | "
            f"{row['labeled']} | {row['completion']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Judgment Count by Query Type",
            "",
            "| Query type | Judgments |",
            "| --- | ---: |",
        ]
    )
    for query_type, count in stats["judgment_count_by_type"].items():
        lines.append(f"| `{query_type}` | {count} |")

    return "\n".join(lines) + "\n"


def compute_progress_stats(judgments: pd.DataFrame) -> dict[str, Any]:
    if judgments.empty:
        return {
            "number_of_queries": 0,
            "total_judgments": 0,
            "labeled_judgments": 0,
            "unlabeled_judgments": 0,
            "completion_percentage": 0.0,
            "fully_labeled_queries": 0,
            "partially_labeled_queries": 0,
            "not_started_queries": 0,
            "query_type_rows": [],
            "judgment_count_by_type": {},
        }

    labeled_mask = judgments["relevance"].astype(str).str.strip() != ""
    total = len(judgments)
    labeled = int(labeled_mask.sum())

    query_statuses = []
    for query_id, group in judgments.assign(_labeled=labeled_mask).groupby("query_id"):
        labeled_count = int(group["_labeled"].sum())
        total_count = len(group)
        query_statuses.append((query_id, total_count, labeled_count))

    fully = sum(1 for _, total_count, labeled_count in query_statuses if labeled_count == total_count)
    partial = sum(1 for _, _, labeled_count in query_statuses if 0 < labeled_count)
    partial -= fully
    not_started = sum(1 for _, _, labeled_count in query_statuses if labeled_count == 0)

    type_rows: list[dict[str, Any]] = []
    for query_type, group in judgments.assign(_labeled=labeled_mask).groupby("type"):
        judgments_count = len(group)
        labeled_count = int(group["_labeled"].sum())
        type_rows.append(
            {
                "type": query_type,
                "queries": int(group["query_id"].nunique()),
                "judgments": judgments_count,
                "labeled": labeled_count,
                "completion": (labeled_count / judgments_count * 100.0) if judgments_count else 0.0,
            }
        )

    return {
        "number_of_queries": int(judgments["query_id"].nunique()),
        "total_judgments": total,
        "labeled_judgments": labeled,
        "unlabeled_judgments": total - labeled,
        "completion_percentage": (labeled / total * 100.0) if total else 0.0,
        "fully_labeled_queries": fully,
        "partially_labeled_queries": partial,
        "not_started_queries": not_started,
        "query_type_rows": sorted(type_rows, key=lambda row: str(row["type"])),
        "judgment_count_by_type": {
            str(query_type): int(count)
            for query_type, count in judgments.groupby("type").size().sort_index().items()
        },
    }


def write_docs(output_dir: Path, judgments: pd.DataFrame) -> None:
    (output_dir / "README.md").write_text(build_readme(), encoding="utf-8")
    (output_dir / "labeling_guide.md").write_text(build_labeling_guide(), encoding="utf-8")
    (output_dir / "presentation_summary.md").write_text(build_presentation_summary(), encoding="utf-8")
    (output_dir / "labeling_progress.md").write_text(build_progress_markdown(judgments), encoding="utf-8")


def build_readme() -> str:
    return """# Relevance Labeling

This experiment stage prepares a deduplicated manual relevance-judgment pool from the ranked validation search results.

No final search-quality metrics are calculated here. Metrics are intentionally left for the next experiment stage after manual labels are complete.

## Input

- `experiments/09_validation_results/all_validation_results.csv`

The input contains ranked top-k outputs from FAISS, Qdrant, filtering, style reranking, and object-aware reranking systems.

## Outputs

| File | Purpose |
| --- | --- |
| `relevance_judgments.csv` | Primary manual labeling file, one row per unique query-image pair. |
| `relevance_judgments_by_query.csv` | Same judgments sorted for query-by-query review. |
| `unlabeled_items.csv` | Rows where `relevance` is still empty. |
| `labeled_items.csv` | Rows where `relevance` has been completed. |
| `labeling_progress.md` | Regenerated progress summary for reporting. |
| `labeling_guide.md` | Human labeling instructions and relevance rules. |
| `presentation_summary.md` | Short slide-ready summary of the labeling method. |

## Duplicate Handling

The same image can appear for the same query in multiple systems. This stage keeps one judgment row per unique `query_id + image_id` pair and stores the contributing systems in `source_systems`.

Each row also preserves the best rank, ranks by system, highest available score, image path, detected objects, matched keywords, and visual descriptor fields where available.

## Label Preservation

When `relevance_judgments.csv` already exists, the preparation script preserves any non-empty `relevance`, `confidence`, and `comment` values by matching on `judgment_id`.

## Workflow

1. Generate validation search results.
2. Prepare unique relevance judgments.
3. Open `relevance_judgments.csv`.
4. Fill `relevance`, `confidence`, and `comment`.
5. Run `check_relevance_labeling.py`.
6. Proceed to the metrics evaluation stage.

## Run Preparation

```powershell
python experiments/scripts/prepare_relevance_labeling.py
```

Optional:

```powershell
python experiments/scripts/prepare_relevance_labeling.py --input-path experiments/09_validation_results/all_validation_results.csv --output-dir experiments/10_relevance_labeling --max-items-per-query 50
```

## Manual Labeling

Fill only these manual fields:

- `relevance`: `2`, `1`, or `0`
- `confidence`: optional `high`, `medium`, or `low`
- `comment`: optional short note

Use `labeling_guide.md` for the detailed rules. Judge visual relevance to the query, not the CLIP score, keyword text, or detector labels alone.

## Validate Labels

```powershell
python experiments/scripts/check_relevance_labeling.py
```
"""


def build_labeling_guide() -> str:
    return """# Relevance Labeling Guide

Human labels should be based primarily on visual relevance to the query. Do not judge a result as relevant only because the CLIP score is high, keywords match, or YOLO detected an object.

## What to Inspect

Open or preview the image referenced by `image_path`, then compare the visible content to the `query` and `type` fields. Use metadata fields only as supporting context.

## Relevance Scale

| Label | Meaning |
| ---: | --- |
| 2 | Highly relevant |
| 1 | Partially relevant |
| 0 | Not relevant |

### 2 = Highly Relevant

The result strongly satisfies the query.

- Correct main object.
- Correct visual scene.
- Correct requested photographic style.
- Correct object and style combination for mixed queries.
- Strong visual similarity for image-to-image queries.

### 1 = Partially Relevant

The result satisfies part of the query but misses an important element.

- Correct object but wrong scene.
- Correct style but weak semantic match.
- Correct scene but missing requested object.
- Visually related but not a strong match.

### 0 = Not Relevant

The result does not meaningfully satisfy the query.

- Wrong object.
- Unrelated scene.
- Opposite visual style.
- Accidental keyword or detector match.
- Visually unrelated image.

## Query-Type Guidance

### Semantic Queries

Judge whether the main scene or subject visually matches the query. Style can help, but semantic content should drive the label.

### Style Queries

Inspect mood, lighting, color, contrast, saturation, and composition. The exact object may be less important unless the query names one.

### Object Queries

Inspect whether the requested object is actually visible. Do not trust YOLO blindly; detector labels are hints, not evidence.

### Mixed Queries

Require both semantic or object relevance and the requested style or scene. A result that satisfies only one side is usually `1`, not `2`.

### Image-to-Image Queries

Judge visual similarity to the query image, including subject, scene, composition, color, lighting, and photographic mood.

## Confidence

Confidence is optional and may remain empty during initial labeling.

- `high`: the label is obvious.
- `medium`: some ambiguity exists.
- `low`: the query or image is difficult to judge.

## Ambiguous Cases

Use `1` when a result is plausibly related but incomplete. Use `confidence=low` and add a short comment when the image is hard to interpret, the query is broad, or metadata conflicts with the visible image.
"""


def build_presentation_summary() -> str:
    return """# Manual Relevance Assessment

- Deduplicated query-image pairs are labeled once, even when returned by multiple systems.
- Labels use a graded relevance scale: 0, 1, and 2.
- Shared labels are reused for fair comparison across retrieval systems.
- Progress tracking separates labeled and unlabeled judgments.

| Label | Meaning |
| ---: | --- |
| 2 | Highly relevant |
| 1 | Partially relevant |
| 0 | Not relevant |

A shared judgment pool ensures that all retrieval systems are evaluated against the same human relevance criteria.
"""


def prepare_relevance_labeling(
    input_path: Path,
    output_dir: Path,
    max_items_per_query: int = DEFAULT_MAX_ITEMS_PER_QUERY,
) -> dict[str, int]:
    input_path = resolve_project_path(input_path)
    output_dir = resolve_project_path(output_dir)

    raw_results = load_validation_results(input_path)
    raw_row_count = len(raw_results)
    judgments = aggregate_judgments(raw_results, max_items_per_query=max_items_per_query)
    previous_labels = load_previous_labels(output_dir / "relevance_judgments.csv")
    judgments = preserve_previous_labels(judgments, previous_labels)

    write_labeling_csvs(judgments, output_dir)
    write_docs(output_dir, judgments)

    return {
        "raw_rows": raw_row_count,
        "unique_judgments": len(judgments),
        "duplicates_removed": raw_row_count - len(judgments),
        "queries": int(judgments["query_id"].nunique()) if not judgments.empty else 0,
        "labeled": int((judgments["relevance"].astype(str).str.strip() != "").sum()) if not judgments.empty else 0,
        "unlabeled": int((judgments["relevance"].astype(str).str.strip() == "").sum()) if not judgments.empty else 0,
    }


def main() -> None:
    args = parse_args()
    summary = prepare_relevance_labeling(
        args.input_path,
        args.output_dir,
        max_items_per_query=args.max_items_per_query,
    )

    output_dir = resolve_project_path(args.output_dir)
    print(f"Input rows: {summary['raw_rows']}")
    print(f"Unique query-image judgments: {summary['unique_judgments']}")
    print(f"Duplicate judgments removed: {summary['duplicates_removed']}")
    print(f"Queries included: {summary['queries']}")
    print(f"Labeled judgments: {summary['labeled']}")
    print(f"Unlabeled judgments: {summary['unlabeled']}")
    print(f"Saved labeling files: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
