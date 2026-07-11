from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.search.qdrant_config import get_qdrant_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an optimized Qdrant collection for vector-memory experiments."
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name. Defaults to QDRANT_COLLECTION or photos_optimized.",
    )
    parser.add_argument("--vector-size", type=int, default=512)
    parser.add_argument(
        "--datatype",
        choices=["float32", "float16", "uint8"],
        default="float32",
        help="Native Qdrant vector datatype.",
    )
    parser.add_argument(
        "--scalar-int8",
        action="store_true",
        help="Enable Qdrant scalar int8 quantization.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Scalar quantization quantile in [0.5, 1.0].",
    )
    parser.add_argument(
        "--always-ram",
        action="store_true",
        help="Keep quantized vectors in RAM. Leave false for lower RAM pressure.",
    )
    parser.add_argument(
        "--on-disk",
        action="store_true",
        help="Store original vectors on disk to reduce RAM usage.",
    )
    return parser.parse_args()


def datatype(value: str) -> models.Datatype:
    return {
        "float32": models.Datatype.FLOAT32,
        "float16": models.Datatype.FLOAT16,
        "uint8": models.Datatype.UINT8,
    }[value]


def create_client() -> QdrantClient:
    settings = get_qdrant_settings()
    if settings.mode == "server":
        return QdrantClient(url=settings.url)
    return QdrantClient(path=settings.path)


def main() -> None:
    args = parse_args()
    if args.vector_size <= 0:
        raise ValueError("--vector-size must be positive")
    if args.scalar_int8 and not 0.5 <= args.quantile <= 1.0:
        raise ValueError("--quantile must be in [0.5, 1.0]")

    collection_name = (
        args.collection
        or os.getenv("QDRANT_COLLECTION", "").strip()
        or "photos_optimized"
    )
    quantization_config = None
    if args.scalar_int8:
        quantization_config = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=args.quantile,
                always_ram=args.always_ram,
            )
        )

    client = create_client()
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=args.vector_size,
                distance=models.Distance.COSINE,
                datatype=datatype(args.datatype),
                quantization_config=quantization_config,
                on_disk=args.on_disk,
            ),
        )
        info = client.get_collection(collection_name)
    finally:
        client.close()

    settings = get_qdrant_settings()
    print(f"Qdrant optimized collection recreated: {collection_name}")
    print(f"Mode: {settings.mode}")
    print(f"Storage: {settings.storage_label}")
    print(f"Vector size: {args.vector_size}")
    print(f"Distance: Cosine")
    print(f"Datatype: {args.datatype}")
    print(f"On disk: {args.on_disk}")
    print(f"Scalar int8 quantization: {args.scalar_int8}")
    if args.scalar_int8:
        print(f"Quantile: {args.quantile}")
        print(f"Always RAM: {args.always_ram}")
    print(f"Collection status: {info.status}")


if __name__ == "__main__":
    main()
