import time
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


PHOTOS_PATH = Path("data/unsplash-lite/photos.csv000")
IMAGES_DIR = Path("data/unsplash-lite/images")
METADATA_PATH = Path("data/unsplash-lite/metadata.csv")

TARGET_IMAGES = 25_000
MIN_SIZE = 256
SLEEP_SECONDS = 0.05


IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def build_download_url(url: str) -> str:
    return f"{url}?w=512&q=80&fm=jpg&fit=max"


def download_image(url: str) -> Image.Image:
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def safe_text(value):
    if pd.isna(value):
        return None
    return str(value)


def safe_int(value):
    if pd.isna(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def safe_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_metadata_row(row, image_path: Path, width: int | None, height: int | None) -> dict:
    photo_id = str(row["photo_id"])
    image_url = row.get("photo_image_url")
    download_url = build_download_url(image_url) if not pd.isna(image_url) else None

    return {
        "image_id": photo_id,
        "file_path": str(image_path),
        "photo_url": safe_text(row.get("photo_url")),
        "photo_image_url": safe_text(image_url),
        "download_url": safe_text(download_url),
        "width": width,
        "height": height,
        "aspect_ratio": safe_float(row.get("photo_aspect_ratio")),
        "description": safe_text(row.get("photo_description")),
        "ai_description": safe_text(row.get("ai_description")),
        "photographer_username": safe_text(row.get("photographer_username")),
        "stats_views": safe_int(row.get("stats_views")),
        "stats_downloads": safe_int(row.get("stats_downloads")),
        "blur_hash": safe_text(row.get("blur_hash")),
        "source": "unsplash-lite",
    }


def get_existing_image_size(image_path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None, None


def main() -> None:
    if not PHOTOS_PATH.exists():
        raise FileNotFoundError(f"Photos file not found: {PHOTOS_PATH}")

    photos = pd.read_csv(PHOTOS_PATH, sep="\t")

    rows = []
    existing_count = 0
    newly_downloaded = 0
    skipped = 0

    for _, row in tqdm(photos.iterrows(), total=len(photos), desc="Preparing images"):
        if len(rows) >= TARGET_IMAGES:
            break

        photo_id = str(row["photo_id"])
        image_url = row.get("photo_image_url")

        if pd.isna(image_url):
            skipped += 1
            continue

        image_path = IMAGES_DIR / f"{photo_id}.jpg"

        if image_path.exists():
            width, height = get_existing_image_size(image_path)

            rows.append(
                build_metadata_row(
                    row=row,
                    image_path=image_path,
                    width=width,
                    height=height,
                )
            )

            existing_count += 1
            continue

        download_url = build_download_url(str(image_url))

        try:
            image = download_image(download_url)
            width, height = image.size

            if width < MIN_SIZE or height < MIN_SIZE:
                skipped += 1
                continue

            image.save(image_path, "JPEG", quality=90)

            rows.append(
                build_metadata_row(
                    row=row,
                    image_path=image_path,
                    width=width,
                    height=height,
                )
            )

            newly_downloaded += 1
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            skipped += 1
            print(f"Skipped {photo_id}: {e}")
            continue

    metadata = pd.DataFrame(rows)
    metadata.to_csv(METADATA_PATH, index=False)

    print()
    print(f"Target images: {TARGET_IMAGES}")
    print(f"Metadata rows: {len(rows)}")
    print(f"Existing images reused: {existing_count}")
    print(f"Newly downloaded: {newly_downloaded}")
    print(f"Skipped: {skipped}")
    print(f"Saved metadata to: {METADATA_PATH}")


if __name__ == "__main__":
    main()