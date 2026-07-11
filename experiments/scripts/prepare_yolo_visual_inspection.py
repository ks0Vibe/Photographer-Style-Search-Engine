from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


STAGE_DIR = PROJECT_ROOT / "experiments" / "06_yolo_object_retrieval"
RETRIEVAL_RESULTS_PATH = STAGE_DIR / "retrieval_results.csv"
TEMPLATE_PATH = STAGE_DIR / "visual_inspection_template.csv"
VISUAL_INSPECTION_PATH = STAGE_DIR / "visual_inspection.csv"
GUIDE_PATH = STAGE_DIR / "manual_visual_inspection_guide.md"

OUTPUT_COLUMNS = [
    "query",
    "query_group",
    "mode",
    "rank",
    "image_id",
    "file_path",
    "requested_object",
    "object_present",
    "visual_relevance",
    "visual_notes",
    "visible_main_object",
    "style_or_context_match",
    "failure_reason",
]

MANUAL_COLUMNS = [
    "object_present",
    "visual_relevance",
    "visual_notes",
    "visible_main_object",
    "style_or_context_match",
    "failure_reason",
]
REQUESTED_OBJECT_BY_QUERY = {
    "person": "person",
    "car": "car",
    "dog": "dog",
    "cat": "cat",
    "building": "building",
    "bird": "bird",
    "person in street photography": "person",
    "car at night": "car",
    "dog on beach": "dog",
    "cat indoors": "cat",
    "bird in nature": "bird",
    "building in city": "building",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("query", "")),
        str(row.get("mode", "")),
        str(row.get("rank", "")),
        str(row.get("image_id", "")),
    )


def manual_values_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, str]]:
    output: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        values = {column: str(row.get(column, "") or "") for column in MANUAL_COLUMNS}
        if any(values.values()):
            output[row_key(row)] = values
    return output


def build_visual_rows() -> list[dict[str, str]]:
    retrieval_rows = read_csv(RETRIEVAL_RESULTS_PATH)
    if not retrieval_rows:
        raise FileNotFoundError(f"No retrieval rows found at {RETRIEVAL_RESULTS_PATH}")

    preserved = manual_values_by_key(read_csv(TEMPLATE_PATH))
    preserved.update(manual_values_by_key(read_csv(VISUAL_INSPECTION_PATH)))

    output_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for source_row in retrieval_rows:
        key = row_key(source_row)
        if key in seen:
            continue
        seen.add(key)

        row = {column: str(source_row.get(column, "") or "") for column in OUTPUT_COLUMNS}
        row["requested_object"] = str(
            source_row.get("requested_object", "") or REQUESTED_OBJECT_BY_QUERY.get(row["query"], "")
        )
        for column in MANUAL_COLUMNS:
            row[column] = preserved.get(key, {}).get(column, "")
        output_rows.append(row)

    return output_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_guide() -> None:
    GUIDE_PATH.write_text(
        "\n".join(
            [
                "# Manual Visual Inspection Guide",
                "",
                "Use `visual_inspection.csv` to label the top-10 grids from the YOLO object retrieval experiment.",
                "",
                "## Label Scale",
                "",
                "- `2` = full visual match",
                "- `1` = partial visual match",
                "- `0` = visual failure",
                "",
                "## Object Precision@10 (required)",
                "",
                "For every result in the `qdrant_semantic`, `qdrant_object`, and `qdrant_object_rerank` rows, fill `object_present` with exactly `1` when the requested object is visibly present and `0` when it is absent. Do not infer this from the YOLO payload or metadata; inspect the image. Leave no blank cells in these three modes for the 12 object queries if you want the aggregate metric.",
                "",
                "The aggregate `object_precision_metrics.csv` reports Object Precision@10 before (`qdrant_semantic`) and after object filter/rerank (`qdrant_object`, `qdrant_object_rerank`).",
                "",
                "## Photo folders",
                "",
                "Run `experiments/scripts/export_yolo_manual_photos.py` to create `manual_photos/<query>/<mode>/` with one image file per ranked result.",
                "",
                "## Object-Like Queries",
                "",
                "- `2` = requested object is clearly visible and central",
                "- `1` = requested object is present but small, ambiguous, or not central",
                "- `0` = requested object is not visible",
                "",
                "## Combined Queries",
                "",
                "- `2` = requested object and requested context are both visible",
                "- `1` = only object or only context is visible",
                "- `0` = neither is visible",
                "",
                "## Failure Reasons",
                "",
                "Use one of these values in `failure_reason`:",
                "",
                "- `good_match`",
                "- `partial_match`",
                "- `wrong_object`",
                "- `wrong_scene`",
                "- `object_too_small`",
                "- `style_or_context_mismatch`",
                "- `yolo_miss`",
                "- `metadata_noise`",
                "- `unsupported_class`",
                "- `too_generic`",
                "",
                "Leave fields empty until you have inspected the corresponding image.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    rows = build_visual_rows()
    write_csv(VISUAL_INSPECTION_PATH, rows)
    write_guide()
    print(f"Wrote visual inspection CSV: {VISUAL_INSPECTION_PATH}")
    print(f"Wrote visual inspection guide: {GUIDE_PATH}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
