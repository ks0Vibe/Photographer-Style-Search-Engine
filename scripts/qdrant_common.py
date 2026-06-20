import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder
from app.search import QdrantRetrievalService, QdrantStore


DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
KEYWORDS_PATH = PROJECT_ROOT / "data" / "unsplash-lite" / "keywords.csv000"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "visualizations"
COLLECTION_NAME = "photos"


def create_qdrant_service() -> QdrantRetrievalService:
    return QdrantRetrievalService(
        clip_encoder=CLIPEncoder(),
        qdrant_store=QdrantStore(
            collection_name=COLLECTION_NAME,
            qdrant_path=QDRANT_PATH,
        ),
    )


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--keyword", type=str, default=None, help="Keyword payload filter.")
    parser.add_argument("--object", dest="object_filter", type=str, default=None, help="Detected object payload filter.")
    parser.add_argument("--rerank", action="store_true", help="Apply style-aware reranking after Qdrant retrieval.")
    parser.add_argument("--candidate-pool-size", type=int, default=100, help="Candidate pool size before reranking.")

    for field in ("brightness", "contrast", "saturation", "warmth"):
        parser.add_argument(
            f"--min-{field}",
            dest=f"min_{field}",
            type=float,
            default=None,
            help=f"Minimum {field} filter.",
        )
        parser.add_argument(
            f"--max-{field}",
            dest=f"max_{field}",
            type=float,
            default=None,
            help=f"Maximum {field} filter.",
        )


def extract_filter_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "keyword_filter": args.keyword,
        "object_filter": getattr(args, "object_filter", None),
        "rerank": getattr(args, "rerank", False),
        "candidate_pool_size": getattr(args, "candidate_pool_size", 100),
        "min_brightness": getattr(args, "min_brightness", None),
        "max_brightness": getattr(args, "max_brightness", None),
        "min_contrast": getattr(args, "min_contrast", None),
        "max_contrast": getattr(args, "max_contrast", None),
        "min_saturation": getattr(args, "min_saturation", None),
        "max_saturation": getattr(args, "max_saturation", None),
        "min_warmth": getattr(args, "min_warmth", None),
        "max_warmth": getattr(args, "max_warmth", None),
    }


def get_query_row(image_id: str | None) -> tuple[str, str]:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        cursor = conn.cursor()
        if image_id:
            cursor.execute(
                """
                SELECT image_id, file_path
                FROM images
                WHERE image_id = ?
                """,
                (image_id,),
            )
        else:
            cursor.execute(
                """
                SELECT image_id, file_path
                FROM images
                ORDER BY image_id
                LIMIT 1
                """
            )
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None:
        label = image_id if image_id else "deterministic query row"
        raise RuntimeError(f"Could not find {label} in the metadata database")

    return str(row[0]), str(row[1])


def resolve_image_path(file_path: str) -> Path:
    path = Path(file_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
