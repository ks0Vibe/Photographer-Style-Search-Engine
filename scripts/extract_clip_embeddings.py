import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder


DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "clip_embeddings.npy"
IMAGE_IDS_PATH = EMBEDDINGS_DIR / "image_ids.npy"


def get_connection() -> sqlite3.Connection:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")
    return sqlite3.connect(DATABASE_PATH)


def load_images(cursor: sqlite3.Cursor) -> list[tuple[str, str]]:
    cursor.execute("""
        SELECT image_id, file_path
        FROM images
        ORDER BY image_id
    """)
    return cursor.fetchall()


def resolve_image_path(file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def extract_embedding(encoder: CLIPEncoder, image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return encoder.encode_image(image.convert("RGB"))


def main() -> None:
    print(f"Database: {DATABASE_PATH}")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        images = load_images(cursor)
    finally:
        conn.close()

    print(f"Found images: {len(images)}")

    if not images:
        raise RuntimeError("No images found in database")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    encoder = CLIPEncoder()

    embeddings: list[np.ndarray] = []
    image_ids: list[str] = []
    failed = 0

    for image_id, file_path in tqdm(images, desc="Extracting CLIP embeddings"):
        image_path = resolve_image_path(file_path)

        try:
            embedding = extract_embedding(encoder, image_path)
        except Exception as exc:
            print(f"Failed {image_id}: {exc}")
            failed += 1
            continue

        embeddings.append(embedding)
        image_ids.append(image_id)

    if not embeddings:
        raise RuntimeError("No embeddings were extracted")

    embedding_matrix = np.stack(embeddings).astype(np.float32, copy=False)
    image_ids_array = np.array(image_ids)

    np.save(EMBEDDINGS_PATH, embedding_matrix)
    np.save(IMAGE_IDS_PATH, image_ids_array)

    print(f"\nSaved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Saved image IDs to: {IMAGE_IDS_PATH}")
    print(f"\nEmbeddings shape: {embedding_matrix.shape}")
    print(f"Image IDs shape: {image_ids_array.shape}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
