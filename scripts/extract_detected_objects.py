from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract YOLO detected object labels into SQLite.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics YOLO model path or name.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of rows to consider.")
    parser.add_argument("--offset", type=int, default=0, help="Number of database rows to skip.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of images per YOLO predict batch.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="Inference device.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess rows even when detected_objects is non-empty.")
    parser.add_argument("--dry-run", action="store_true", help="Read rows and resolve files without running YOLO or writing SQLite.")
    return parser.parse_args()


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Missing dependency: ultralytics")
        print("Install command:")
        print(r".\.venv\Scripts\python.exe -m pip install -r requirements.txt")
        raise SystemExit(2)
    return YOLO


def ensure_schema(conn: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(images);").fetchall()}
    missing = [column for column in ("detected_objects", "detection_model", "detection_updated_at") if column not in columns]
    if missing:
        raise RuntimeError(
            "Detection columns are missing. Run scripts/migrate_add_detected_objects.py first. "
            f"Missing: {', '.join(missing)}"
        )


def load_rows(conn: sqlite3.Connection, limit: int | None, offset: int) -> list[sqlite3.Row]:
    query = """
        SELECT image_id, file_path, detected_objects
        FROM images
        ORDER BY image_id
        LIMIT ? OFFSET ?
    """
    effective_limit = limit if limit is not None else -1
    return list(conn.execute(query, (effective_limit, offset)).fetchall())


def parse_detected_objects(raw: Any) -> list[str]:
    if raw is None:
        return []
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip().lower() for item in value if str(item).strip()})


def resolve_image_path(file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def batched(items: list[tuple[str, Path]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def labels_from_result(result) -> list[str]:
    names = result.names or {}
    labels: set[str] = set()
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return []

    for cls_value in boxes.cls.tolist():
        label = str(names.get(int(cls_value), str(int(cls_value)))).strip().lower()
        if label:
            labels.add(label)
    return sorted(labels)


def update_detection(
    conn: sqlite3.Connection,
    image_id: str,
    labels: list[str],
    model_name: str,
    updated_at: str,
) -> None:
    conn.execute(
        """
        UPDATE images
        SET detected_objects = ?,
            detection_model = ?,
            detection_updated_at = ?
        WHERE image_id = ?
        """,
        (json.dumps(labels), model_name, updated_at, image_id),
    )


def main() -> None:
    args = parse_args()
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    start_time = time.perf_counter()
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        ensure_schema(conn)
        rows = load_rows(conn, limit=args.limit, offset=args.offset)

        skipped_existing = 0
        missing_files = 0
        candidates: list[tuple[str, Path]] = []

        for row in rows:
            image_id = str(row["image_id"])
            existing = parse_detected_objects(row["detected_objects"])
            if existing and not args.overwrite:
                skipped_existing += 1
                continue

            image_path = resolve_image_path(str(row["file_path"]))
            if not image_path.exists():
                missing_files += 1
                continue
            candidates.append((image_id, image_path))

        if args.dry_run:
            print("Dry run complete.")
            print(f"candidate_rows={len(rows)}")
            print(f"would_process_images={len(candidates)}")
            print(f"skipped_existing={skipped_existing}")
            print(f"missing_files={missing_files}")
            return

        YOLO = require_ultralytics()
        model = YOLO(args.model)
        device = None if args.device == "auto" else args.device
        updated_at = datetime.now(timezone.utc).isoformat()
        object_counter: Counter[str] = Counter()
        processed_images = 0
        images_with_objects = 0

        for batch in tqdm(list(batched(candidates, args.batch_size)), desc="YOLO detection"):
            paths = [str(path) for _, path in batch]
            results = model.predict(
                paths,
                conf=args.confidence,
                iou=args.iou,
                device=device,
                verbose=False,
            )

            for (image_id, _), result in zip(batch, results, strict=True):
                labels = labels_from_result(result)
                update_detection(conn, image_id, labels, args.model, updated_at)
                processed_images += 1
                if labels:
                    images_with_objects += 1
                    object_counter.update(labels)

            conn.commit()

        elapsed_seconds = time.perf_counter() - start_time
        coverage = images_with_objects / processed_images if processed_images else 0.0
        print()
        print(f"processed_images={processed_images}")
        print(f"skipped_existing={skipped_existing}")
        print(f"missing_files={missing_files}")
        print(f"images_with_objects={images_with_objects}")
        print(f"object_coverage_in_processed_subset={coverage:.4f}")
        print("top_detected_objects:")
        for label, count in object_counter.most_common(20):
            print(f"- {label}: {count}")
        print(f"elapsed_seconds={elapsed_seconds:.2f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
