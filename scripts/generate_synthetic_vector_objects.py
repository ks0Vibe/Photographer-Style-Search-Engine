from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.search import load_keywords_by_image_id


EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"
KEYWORDS_PATH = PROJECT_ROOT / "data" / "unsplash-lite" / "keywords.csv000"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_500k"
GENERATION_NAME = "clip_embedding_perturbation_v1"
PAYLOAD_FIELDS = (
    "source_image_id",
    "file_path",
    "photo_url",
    "ai_description",
    "photographer_username",
    "keywords",
    "detected_objects",
    "brightness",
    "contrast",
    "saturation",
    "warmth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CLIP vector objects from the real embedding distribution."
    )
    parser.add_argument("--target-count", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--mix-neighbor-prob", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_args(args: argparse.Namespace) -> None:
    if args.target_count <= 0:
        raise ValueError("--target-count must be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.noise_std < 0:
        raise ValueError("--noise-std must be non-negative")
    if not 0 <= args.mix_neighbor_prob <= 1:
        raise ValueError("--mix-neighbor-prob must be between 0 and 1")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
    if embeddings.shape[1] != 512:
        raise ValueError(f"Expected embedding shape (N, 512), got {embeddings.shape}")

    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Embeddings contain zero-length vectors")
    return matrix / norms


def parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return sorted({str(item).strip().lower() for item in parsed if str(item).strip()})


def load_metadata_by_id(database_path: Path) -> dict[str, dict[str, Any]]:
    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(images);").fetchall()}
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
                {detected_objects_select}
            FROM images
            """
        ).fetchall()
    finally:
        conn.close()

    return {str(row["image_id"]): dict(row) for row in rows}


def load_payload_templates(image_ids: np.ndarray) -> list[dict[str, Any]]:
    metadata_by_id = load_metadata_by_id(DATABASE_PATH)
    keywords_by_image_id = load_keywords_by_image_id(KEYWORDS_PATH) if KEYWORDS_PATH.exists() else {}
    templates: list[dict[str, Any]] = []

    for raw_image_id in image_ids.tolist():
        image_id = str(raw_image_id)
        row = metadata_by_id.get(image_id)
        if row is None:
            raise RuntimeError(f"Metadata row missing for image_id={image_id}")

        templates.append(
            {
                "source_image_id": image_id,
                "file_path": row.get("file_path"),
                "photo_url": row.get("photo_url"),
                "ai_description": row.get("ai_description"),
                "photographer_username": row.get("photographer_username"),
                "keywords": keywords_by_image_id.get(image_id, []),
                "detected_objects": parse_json_list(row.get("detected_objects")),
                "brightness": to_optional_float(row.get("brightness")),
                "contrast": to_optional_float(row.get("contrast")),
                "saturation": to_optional_float(row.get("saturation")),
                "warmth": to_optional_float(row.get("warmth")),
            }
        )

    return templates


def to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_output_paths(output_dir: Path, overwrite: bool) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "embeddings": output_dir / "synthetic_embeddings.npy",
        "image_ids": output_dir / "synthetic_image_ids.npy",
        "payloads": output_dir / "synthetic_payloads.jsonl",
        "stats": output_dir / "synthetic_generation_stats.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {joined}. Pass --overwrite to replace.")
    for path in existing:
        path.unlink()
    return paths


def synthetic_id(index: int) -> str:
    return f"syn_{index + 1:012d}"


def json_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in payload.items()
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_output_dir(args.output_dir)
    paths = ensure_output_paths(output_dir, overwrite=args.overwrite)

    real_embeddings_raw = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    real_image_ids = np.load(IMAGE_IDS_PATH, allow_pickle=False)
    if real_embeddings_raw.shape[0] != real_image_ids.shape[0]:
        raise ValueError(
            "Embedding and image ID counts do not match: "
            f"{real_embeddings_raw.shape[0]} != {real_image_ids.shape[0]}"
        )

    real_embeddings = normalize_embeddings(real_embeddings_raw)
    real_count, embedding_dim = real_embeddings.shape
    templates = load_payload_templates(real_image_ids)

    rng = np.random.default_rng(args.seed)
    synthetic_embeddings = np.lib.format.open_memmap(
        paths["embeddings"],
        mode="w+",
        dtype=np.float32,
        shape=(args.target_count, embedding_dim),
    )
    synthetic_image_ids = np.lib.format.open_memmap(
        paths["image_ids"],
        mode="w+",
        dtype="<U16",
        shape=(args.target_count,),
    )

    keyword_counter: Counter[str] = Counter()
    object_counter: Counter[str] = Counter()

    with paths["payloads"].open("w", encoding="utf-8") as payload_handle:
        for start in range(0, args.target_count, args.batch_size):
            end = min(start + args.batch_size, args.target_count)
            batch_count = end - start
            source_indices = rng.integers(0, real_count, size=batch_count)
            neighbor_indices = rng.integers(0, real_count, size=batch_count)
            use_neighbor = rng.random(batch_count) < args.mix_neighbor_prob
            noise = rng.normal(0.0, 1.0, size=(batch_count, embedding_dim)).astype(np.float32)

            source_vectors = real_embeddings[source_indices]
            vectors = source_vectors + args.noise_std * noise
            if np.any(use_neighbor):
                mixed = (
                    0.90 * source_vectors[use_neighbor]
                    + 0.08 * real_embeddings[neighbor_indices[use_neighbor]]
                    + 0.02 * noise[use_neighbor]
                )
                vectors[use_neighbor] = mixed

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = (vectors / norms).astype(np.float32, copy=False)
            synthetic_embeddings[start:end] = vectors

            for batch_offset, source_index in enumerate(source_indices.tolist()):
                global_index = start + batch_offset
                image_id = synthetic_id(global_index)
                synthetic_image_ids[global_index] = image_id

                template = templates[source_index]
                keywords = list(template["keywords"])
                detected_objects = list(template["detected_objects"])
                keyword_counter.update(keywords)
                object_counter.update(detected_objects)

                payload = {
                    **{field: template[field] for field in PAYLOAD_FIELDS},
                    "image_id": image_id,
                    "is_synthetic": True,
                    "synthetic_generation": GENERATION_NAME,
                    "synthetic_noise_std": float(args.noise_std),
                }
                payload_handle.write(json.dumps(json_safe_payload(payload), ensure_ascii=False) + "\n")

            print(f"Generated synthetic vectors: {end}/{args.target_count}", flush=True)

    synthetic_embeddings.flush()
    synthetic_image_ids.flush()
    norm_sample = np.asarray(synthetic_embeddings[: min(args.target_count, 100_000)])
    norms = np.linalg.norm(norm_sample, axis=1)
    output_size_mb = sum(path.stat().st_size for path in paths.values() if path.exists()) / (1024 * 1024)

    stats = {
        "real_count": int(real_count),
        "synthetic_count": int(args.target_count),
        "embedding_dim": int(embedding_dim),
        "synthetic_embeddings_shape": [int(args.target_count), int(embedding_dim)],
        "vector_dtype": "float32",
        "seed": int(args.seed),
        "noise_std": float(args.noise_std),
        "mix_neighbor_prob": float(args.mix_neighbor_prob),
        "mean_norm": float(norms.mean()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "output_size_mb": float(output_size_mb),
        "top_copied_keywords": [
            {"keyword": keyword, "count": int(count)}
            for keyword, count in keyword_counter.most_common(25)
        ],
        "top_copied_detected_objects": [
            {"detected_object": obj, "count": int(count)}
            for obj, count in object_counter.most_common(25)
        ],
        "files": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in paths.items()},
        "note": "Synthetic file_path values are proxy references to source real images for debugging only.",
    }
    paths["stats"].write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    args = parse_args()
    validate_args(args)
    stats = generate(args)

    print("Synthetic generation complete")
    for key in (
        "real_count",
        "synthetic_count",
        "embedding_dim",
        "synthetic_embeddings_shape",
        "mean_norm",
        "min_norm",
        "max_norm",
        "output_size_mb",
        "top_copied_keywords",
        "top_copied_detected_objects",
    ):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
