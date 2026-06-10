import pandas as pd
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

PHOTOS_PATH = Path("data/unsplash-lite/photos.csv000")
IMAGES_DIR = Path("data/unsplash-lite/images")
METADATA_PATH = Path("data/unsplash-lite/metadata.csv")

TARGET_IMAGES = 1000

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def build_download_url(url: str) -> str:
    return f"{url}?w=512&q=80&fm=jpg&fit=max"


def download_image(url: str) -> Image.Image:
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def main():
    photos = pd.read_csv(PHOTOS_PATH, sep="\t")

    rows = []
    downloaded = 0

    for _, row in tqdm(photos.iterrows(), total=len(photos)):
        if downloaded >= TARGET_IMAGES:
            break

        photo_id = row["photo_id"]
        image_url = row["photo_image_url"]

        if pd.isna(image_url):
            continue

        download_url = build_download_url(image_url)
        image_path = IMAGES_DIR / f"{photo_id}.jpg"

        if image_path.exists():
            downloaded += 1
            continue

        try:
            image = download_image(download_url)
            width, height = image.size

            if width < 256 or height < 256:
                continue

            image.save(image_path, "JPEG", quality=90)

            rows.append({
                "image_id": photo_id,
                "file_path": str(image_path),
                "photo_url": row.get("photo_url"),
                "photo_image_url": image_url,
                "download_url": download_url,
                "width": width,
                "height": height,
                "aspect_ratio": row.get("photo_aspect_ratio"),
                "description": row.get("photo_description"),
                "ai_description": row.get("ai_description"),
                "photographer_username": row.get("photographer_username"),
                "stats_views": row.get("stats_views"),
                "stats_downloads": row.get("stats_downloads"),
                "blur_hash": row.get("blur_hash"),
                "source": "unsplash-lite",
            })

            downloaded += 1
            time.sleep(0.05)

        except Exception as e:
            print(f"Skipped {photo_id}: {e}")
            continue

    metadata = pd.DataFrame(rows)
    metadata.to_csv(METADATA_PATH, index=False)

    print(f"Downloaded: {downloaded}")
    print(f"Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()