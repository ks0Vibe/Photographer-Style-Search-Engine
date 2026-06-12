import sys
import sqlite3
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder


DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"


def main():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, file_path FROM images LIMIT 1")
    row = cursor.fetchone()

    conn.close()

    if row is None:
        raise RuntimeError("No images found in database")

    image_id, file_path = row
    image_path = PROJECT_ROOT / file_path

    print(f"Testing image: {image_id}")
    print(f"Path: {image_path}")

    image = Image.open(image_path).convert("RGB")

    encoder = CLIPEncoder()

    image_embedding = encoder.encode_image(image)
    text_embedding = encoder.encode_text("warm cinematic landscape photo")

    print("Image embedding shape:", image_embedding.shape)
    print("Text embedding shape:", text_embedding.shape)

    print("Image embedding norm:", (image_embedding ** 2).sum() ** 0.5)
    print("Text embedding norm:", (text_embedding ** 2).sum() ** 0.5)


if __name__ == "__main__":
    main()