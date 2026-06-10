from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PHOTOS_PATH = PROJECT_ROOT / "data" / "unsplash-lite" / "photos.csv000"
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"


def main():
    if not PHOTOS_PATH.exists():
        raise FileNotFoundError(f"Photos file not found: {PHOTOS_PATH}")

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    photos = pd.read_csv(PHOTOS_PATH, sep="\t")

    image_files = {path.stem: path for path in IMAGES_DIR.glob("*.jpg")}

    if not image_files:
        raise RuntimeError(f"No jpg images found in {IMAGES_DIR}")

    rows = []

    for _, row in photos.iterrows():
        photo_id = str(row["photo_id"])

        if photo_id not in image_files:
            continue

        image_path = image_files[photo_id]

        rows.append({
            "image_id": photo_id,
            "file_path": str(image_path.relative_to(PROJECT_ROOT)),
            "photo_url": row.get("photo_url"),
            "photo_image_url": row.get("photo_image_url"),
            "download_url": f"{row.get('photo_image_url')}?w=512&q=80&fm=jpg&fit=max",
            "width": row.get("photo_width"),
            "height": row.get("photo_height"),
            "aspect_ratio": row.get("photo_aspect_ratio"),
            "description": row.get("photo_description"),
            "ai_description": row.get("ai_description"),
            "photographer_username": row.get("photographer_username"),
            "stats_views": row.get("stats_views"),
            "stats_downloads": row.get("stats_downloads"),
            "blur_hash": row.get("blur_hash"),
            "source": "unsplash-lite",
        })

    metadata = pd.DataFrame(rows)
    metadata.to_csv(METADATA_PATH, index=False)

    print(f"Found images: {len(image_files)}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()