import sqlite3
import sys
from pathlib import Path
from typing import Any
import json

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.search import QdrantStore, VectorStore, load_keywords_by_image_id
from scripts.qdrant_common import (
    COLLECTION_NAME,
    DATABASE_PATH,
    EMBEDDINGS_PATH,
    IMAGE_IDS_PATH,
    KEYWORDS_PATH,
    QDRANT_PATH,
)


def load_metadata_rows(database_path: Path) -> dict[str, dict[str, Any]]:
    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    try:
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(images);").fetchall()
        }
        detected_objects_select = (
            "detected_objects" if "detected_objects" in columns else "'[]' AS detected_objects"
        )
        rows = conn.execute(
            f"""
            SELECT
                image_id,
                file_path,
                photo_url,
                ai_description,
                photographer_username,
                brightness,
                contrast,
                saturation,
                warmth,
                color_histogram,
                {detected_objects_select}
            FROM images
            """
        ).fetchall()
    finally:
        conn.close()

    return {str(row["image_id"]): dict(row) for row in rows}


def build_payloads(
    image_ids: np.ndarray,
    metadata_by_id: dict[str, dict[str, Any]],
    keywords_by_image_id: dict[str, list[str]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    for raw_image_id in image_ids.tolist():
        image_id = str(raw_image_id)
        row = metadata_by_id.get(image_id)
        if row is None:
            raise RuntimeError(f"Metadata row missing for image_id={image_id}")

        payloads.append(
            {
                "image_id": image_id,
                "file_path": row["file_path"],
                "photo_url": row["photo_url"],
                "ai_description": row["ai_description"],
                "photographer_username": row["photographer_username"],
                "brightness": row["brightness"],
                "contrast": row["contrast"],
                "saturation": row["saturation"],
                "warmth": row["warmth"],
                "color_histogram": _parse_histogram(row["color_histogram"]),
                "keywords": keywords_by_image_id.get(image_id, []),
                "detected_objects": _parse_detected_objects(row["detected_objects"]),
            }
        )

    return payloads


def _parse_histogram(raw_histogram: str | None) -> list[float] | None:
    if raw_histogram is None:
        return None

    values = json.loads(raw_histogram)
    return [float(value) for value in values]


def _parse_detected_objects(raw_detected_objects: Any) -> list[str]:
    if raw_detected_objects is None:
        return []

    try:
        values = json.loads(str(raw_detected_objects))
    except json.JSONDecodeError:
        return []

    if not isinstance(values, list):
        return []

    return sorted(
        {
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        }
    )


def main() -> None:
    vector_store = VectorStore(
        embeddings_path=EMBEDDINGS_PATH,
        image_ids_path=IMAGE_IDS_PATH,
    )
    embeddings = vector_store.get_embeddings()
    image_ids = vector_store.get_image_ids()

    metadata_by_id = load_metadata_rows(DATABASE_PATH)
    keywords_by_image_id = load_keywords_by_image_id(KEYWORDS_PATH) if KEYWORDS_PATH.exists() else {}
    payloads = build_payloads(image_ids, metadata_by_id, keywords_by_image_id)

    store = QdrantStore(collection_name=COLLECTION_NAME, qdrant_path=QDRANT_PATH)
    try:
        store.validate_collection_exists()
        store.upload_points(embeddings=embeddings, image_ids=image_ids, payloads=payloads)
        point_count = store.count()
    finally:
        store.close()

    print(f"Uploaded points: {point_count}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Storage: {QDRANT_PATH}")


if __name__ == "__main__":
    main()
