from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "experiments" / "10_relevance_labeling" / "relevance_judgments.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "10_relevance_labeling" / "review_images_by_query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export relevance-judgment images into one review directory per query."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only-unlabeled",
        action="store_true",
        help="Export only rows where relevance is empty.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="Copy image files. This is the default.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sanitize(value: str, max_length: int = 72) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return (cleaned or "untitled")[:max_length]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_rank(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("best_rank", "") or "9999"))
    except ValueError:
        return 9999


def query_directory_name(row: dict[str, str]) -> str:
    query_id = row.get("query_id", "").strip() or "Q_UNKNOWN"
    query = sanitize(row.get("query", ""), max_length=60)
    return f"{query_id}_{query}"


def destination_file_name(index: int, row: dict[str, str], source_path: Path) -> str:
    relevance = row.get("relevance", "").strip()
    rel_label = f"rel{relevance}" if relevance else "rel_unlabeled"
    rank = numeric_rank(row)
    image_id = sanitize(row.get("image_id", ""), max_length=32)
    suffix = source_path.suffix.lower() or ".jpg"
    return f"{index:03d}_{rel_label}_rank{rank:02d}__{image_id}{suffix}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    rows = read_rows(input_path)

    if args.only_unlabeled:
        rows = [row for row in rows if not row.get("relevance", "").strip()]

    rows.sort(key=lambda row: (row.get("query_id", ""), numeric_rank(row), row.get("image_id", "")))
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    index_rows: list[dict[str, Any]] = []
    per_query_counts: dict[str, int] = {}

    for row in rows:
        query_dir_name = query_directory_name(row)
        query_dir = output_dir / query_dir_name
        query_dir.mkdir(parents=True, exist_ok=True)
        per_query_counts[query_dir_name] = per_query_counts.get(query_dir_name, 0) + 1

        source_path = resolve_path(Path(row.get("image_path", "")))
        dest_name = destination_file_name(per_query_counts[query_dir_name], row, source_path)
        dest_path = query_dir / dest_name
        status = "copied"
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            copied += 1
        else:
            status = "missing_source"
            missing += 1

        index_rows.append(
            {
                "query_id": row.get("query_id", ""),
                "query": row.get("query", ""),
                "type": row.get("type", ""),
                "folder": query_dir_name,
                "review_file": str(dest_path.relative_to(output_dir)) if source_path.exists() else "",
                "image_id": row.get("image_id", ""),
                "source_path": row.get("image_path", ""),
                "best_rank": row.get("best_rank", ""),
                "source_systems": row.get("source_systems", ""),
                "relevance": row.get("relevance", ""),
                "confidence": row.get("confidence", ""),
                "status": status,
            }
        )

    write_csv(output_dir / "_index.csv", index_rows)
    missing_rows = [row for row in index_rows if row["status"] == "missing_source"]
    write_csv(output_dir / "_missing.csv", missing_rows)

    readme_lines = [
        "# Relevance Review Images",
        "",
        f"- Source CSV: `{input_path.relative_to(PROJECT_ROOT)}`",
        f"- Exported rows: {len(rows)}",
        f"- Copied images: {copied}",
        f"- Missing source images: {missing}",
        f"- Query folders: {len(per_query_counts)}",
        "",
        "File naming:",
        "",
        "`001_rel2_rank01__imageid.jpg` means local item number 1, relevance label 2, best rank 1.",
        "",
        "Use `_index.csv` to map exported files back to `judgment_id` context through query/image metadata.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Export directory: {output_dir}")
    print(f"Exported rows: {len(rows)}")
    print(f"Copied images: {copied}")
    print(f"Missing source images: {missing}")
    print(f"Query folders: {len(per_query_counts)}")
    if missing:
        print(f"Missing list: {output_dir / '_missing.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
