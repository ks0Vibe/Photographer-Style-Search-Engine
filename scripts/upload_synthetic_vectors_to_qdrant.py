from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from qdrant_client import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.synthetic_qdrant_common import (
    SERVER_QDRANT_STORAGE_PATH,
    SYNTHETIC_DATA_DIR,
    SYNTHETIC_STAGE_DIR,
    create_synthetic_qdrant_store,
    directory_size_mb,
    synthetic_collection_name,
    synthetic_qdrant_mode,
    synthetic_qdrant_path,
    synthetic_storage_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload synthetic vectors to the synthetic Qdrant collection.")
    parser.add_argument("--data-dir", type=Path, default=SYNTHETIC_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--stats-path", type=Path, default=SYNTHETIC_STAGE_DIR / "synthetic_upload_stats.json")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_payloads(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON payload at line {line_number}: {path}") from exc


def batched(iterator: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    data_dir = resolve_path(args.data_dir)
    embeddings_path = data_dir / "synthetic_embeddings.npy"
    image_ids_path = data_dir / "synthetic_image_ids.npy"
    payloads_path = data_dir / "synthetic_payloads.jsonl"
    stats_path = resolve_path(args.stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    image_ids = np.load(image_ids_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Expected synthetic embeddings shape (N, 512), got {embeddings.shape}")
    if embeddings.shape[0] != image_ids.shape[0]:
        raise ValueError(f"Embedding and image ID counts differ: {embeddings.shape[0]} != {image_ids.shape[0]}")

    store = create_synthetic_qdrant_store()
    start_time = time.perf_counter()
    uploaded = 0
    try:
        store.validate_collection_exists()
        payload_iterator = read_payloads(payloads_path)
        for batch_index, payload_batch in enumerate(batched(payload_iterator, args.batch_size)):
            start = batch_index * args.batch_size
            end = start + len(payload_batch)
            vector_batch = np.asarray(embeddings[start:end], dtype=np.float32)
            id_batch = image_ids[start:end].tolist()
            if len(payload_batch) != vector_batch.shape[0]:
                raise ValueError("Payload count and vector batch count do not match")

            points = [
                models.PointStruct(
                    id=start + offset,
                    vector=vector_batch[offset].tolist(),
                    payload={
                        **payload,
                        "image_id": str(id_batch[offset]),
                    },
                )
                for offset, payload in enumerate(payload_batch)
            ]
            store.client.upsert(
                collection_name=synthetic_collection_name(),
                points=points,
                wait=True,
            )
            uploaded = end
            elapsed = time.perf_counter() - start_time
            rate = uploaded / elapsed if elapsed > 0 else 0.0
            print(f"Uploaded synthetic points: {uploaded}/{embeddings.shape[0]} ({rate:.1f} points/sec)", flush=True)

        if uploaded != embeddings.shape[0]:
            raise ValueError(
                "Payload count does not match synthetic embedding count: "
                f"{uploaded} != {embeddings.shape[0]}"
            )
        point_count = store.count()
    finally:
        store.close()

    upload_time = time.perf_counter() - start_time
    storage_size_mb = directory_size_mb(
        synthetic_qdrant_path()
        if synthetic_qdrant_mode() == "local"
        else SERVER_QDRANT_STORAGE_PATH
    )
    stats = {
        "collection_name": synthetic_collection_name(),
        "qdrant_mode": synthetic_qdrant_mode(),
        "storage": synthetic_storage_label(),
        "uploaded_points": int(uploaded),
        "collection_size": int(point_count),
        "embedding_dim": int(embeddings.shape[1]),
        "batch_size": int(args.batch_size),
        "upload_time_seconds": float(upload_time),
        "points_per_second": float(uploaded / upload_time if upload_time > 0 else 0.0),
        "storage_size_mb": storage_size_mb,
        "server_mode_preferred": True,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Upload complete: {uploaded} points")
    print(f"Collection count: {point_count}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
