from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"


def parse_objects(raw: Any) -> list[str]:
    if raw is None:
        return []
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip().lower() for item in value if str(item).strip()})


def main() -> None:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")

    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(images);").fetchall()}
        if "detected_objects" not in columns:
            raise RuntimeError("images.detected_objects does not exist. Run scripts/migrate_add_detected_objects.py first.")

        rows = conn.execute(
            """
            SELECT image_id, file_path, ai_description, detected_objects
            FROM images
            ORDER BY image_id
            """
        ).fetchall()
    finally:
        conn.close()

    counter: Counter[str] = Counter()
    with_objects = []
    without_objects = []

    for row in rows:
        labels = parse_objects(row["detected_objects"])
        if labels:
            with_objects.append((row, labels))
            counter.update(labels)
        elif len(without_objects) < 10:
            without_objects.append(row)

    total_images = len(rows)
    object_count = len(with_objects)
    coverage = object_count / total_images if total_images else 0.0

    print(f"total_images={total_images}")
    print(f"images_with_non_empty_detected_objects={object_count}")
    print(f"object_coverage={coverage:.4f}")
    print(f"unique_detected_objects={len(counter)}")
    print()
    print("top_50_detected_objects:")
    for label, count in counter.most_common(50):
        print(f"- {label}: {count}")
    print()
    print("sample_images_with_objects:")
    for row, labels in with_objects[:10]:
        print(f"- {row['image_id']} objects={labels} path={row['file_path']}")
    print()
    print("sample_images_without_objects:")
    for row in without_objects[:10]:
        print(f"- {row['image_id']} path={row['file_path']}")


if __name__ == "__main__":
    main()
