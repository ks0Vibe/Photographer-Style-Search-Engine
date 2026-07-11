from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = PROJECT_ROOT / "experiments" / "06_yolo_object_retrieval"
DEFAULT_INPUT = STAGE_DIR / "visual_inspection.csv"
DEFAULT_OUTPUT = STAGE_DIR / "manual_photos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO retrieval photos into query-specific folders for manual inspection."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Optional retrieval modes to export. By default all modes in visual_inspection.csv are exported.",
    )
    parser.add_argument(
        "--copy",
        choices=["copy", "hardlink"],
        default="copy",
        help="Use regular copies by default; hardlink is useful when source and destination share a filesystem.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ._-]+", "_", str(value).strip())
    return normalized.strip("._") or "unnamed"


def resolve_image_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_filename(rank: str, image_id: str, source: Path) -> str:
    suffix = source.suffix.lower() or ".jpg"
    try:
        rank_label = f"rank_{int(rank):02d}"
    except ValueError:
        rank_label = f"rank_{safe_name(rank)}"
    return f"{rank_label}_{safe_name(image_id)}{suffix}"


def export(args: argparse.Namespace) -> tuple[int, int, list[str]]:
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Inspection CSV not found: {input_path}")

    selected_modes = set(args.modes or [])
    copied = 0
    missing = 0
    queries: set[str] = set()

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"query", "mode", "rank", "image_id", "file_path"}
        missing_columns = required - set(rows.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing columns in {input_path}: {sorted(missing_columns)}")

        for row in rows:
            query = str(row.get("query") or "").strip()
            mode = str(row.get("mode") or "").strip()
            rank = str(row.get("rank") or "").strip()
            image_id = str(row.get("image_id") or "").strip()
            source = resolve_image_path(str(row.get("file_path") or "").strip())
            if not query or not mode or not rank or not image_id:
                continue
            if selected_modes and mode not in selected_modes:
                continue

            queries.add(query)
            query_dir = output_path / safe_name(query) / safe_name(mode)
            query_dir.mkdir(parents=True, exist_ok=True)
            destination = query_dir / output_filename(rank, image_id, source)

            if not source.exists():
                missing += 1
                continue

            if args.copy == "hardlink":
                if destination.exists():
                    destination.unlink()
                try:
                    destination.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, destination)
            else:
                shutil.copy2(source, destination)
            copied += 1

    return copied, missing, sorted(queries)


def main() -> None:
    args = parse_args()
    copied, missing, queries = export(args)
    print(f"Exported photos: {copied}")
    print(f"Missing source photos: {missing}")
    print(f"Queries: {len(queries)}")
    print(f"Output directory: {resolve_path(args.output)}")
    if missing:
        print("Some rows were skipped because their source file_path does not exist.")


if __name__ == "__main__":
    main()
