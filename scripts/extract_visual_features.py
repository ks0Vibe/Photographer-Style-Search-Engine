import sys
import sqlite3
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from app.ml.visual_descriptor import VisualDescriptorExtractor



DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"


def get_connection():
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")
    return sqlite3.connect(DATABASE_PATH)


def load_images(cursor):
    cursor.execute("SELECT image_id, file_path FROM images")
    return cursor.fetchall()


def update_features(cursor, image_id, features):
    cursor.execute("""
        UPDATE images
        SET brightness = ?,
            contrast = ?,
            saturation = ?,
            warmth = ?,
            color_histogram = ?
        WHERE image_id = ?
    """, (
        features["brightness"],
        features["contrast"],
        features["saturation"],
        features["warmth"],
        json.dumps(features["color_histogram"]),
        image_id,
    ))


def resolve_image_path(file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    extractor = VisualDescriptorExtractor()

    conn = get_connection()
    cursor = conn.cursor()

    images = load_images(cursor)
    print(f"Found images: {len(images)}")

    updated = 0
    failed = 0

    for image_id, file_path in tqdm(images, desc="Extracting visual features"):
        image_path = resolve_image_path(file_path)

        try:
            image = Image.open(image_path).convert("RGB")
            features = extractor.extract(image)
            update_features(cursor, image_id, features)
            updated += 1
        except Exception as e:
            print(f"Failed {image_id}: {e}")
            failed += 1

    conn.commit()
    conn.close()

    print(f"Updated: {updated}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()