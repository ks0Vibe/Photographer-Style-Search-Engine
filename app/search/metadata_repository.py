from __future__ import annotations

import logging
import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageMetadata:
    image_id: str
    file_path: str
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    warmth: float | None = None
    color_histogram: tuple[float, ...] | None = None

    @property
    def has_style_features(self) -> bool:
        return all(
            value is not None
            for value in (
                self.brightness,
                self.contrast,
                self.saturation,
                self.warmth,
                self.color_histogram,
            )
        )


class MetadataRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

    def get_by_id(self, image_id: str) -> ImageMetadata | None:
        if not image_id:
            raise ValueError("image_id must not be empty")

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT image_id, file_path, brightness, contrast, saturation, warmth, color_histogram
                FROM images
                WHERE image_id = ?
                """,
                (image_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_metadata(row)

    def get_many(self, image_ids: list[str]) -> dict[str, ImageMetadata]:
        if not image_ids:
            return {}

        placeholders = ", ".join("?" for _ in image_ids)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT image_id, file_path, brightness, contrast, saturation, warmth, color_histogram
                FROM images
                WHERE image_id IN ({placeholders})
                """,
                tuple(image_ids),
            ).fetchall()

        metadata_by_id = {
            row["image_id"]: self._row_to_metadata(row)
            for row in rows
        }
        logger.info("Loaded metadata rows: requested=%s found=%s", len(image_ids), len(metadata_by_id))
        return metadata_by_id

    def get_style_features(self, image_id: str) -> dict[str, float | list[float]] | None:
        metadata = self.get_by_id(image_id)
        if metadata is None:
            return None
        return self._metadata_to_style_features(metadata)

    def get_style_features_batch(
        self,
        image_ids: list[str],
    ) -> dict[str, dict[str, float | list[float]]]:
        metadata_by_id = self.get_many(image_ids)
        return {
            image_id: self._metadata_to_style_features(metadata)
            for image_id, metadata in metadata_by_id.items()
            if metadata.has_style_features
        }

    def _connect(self) -> sqlite3.Connection:
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.database_path}")

        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_metadata(row: sqlite3.Row) -> ImageMetadata:
        return ImageMetadata(
            image_id=row["image_id"],
            file_path=row["file_path"],
            brightness=row["brightness"],
            contrast=row["contrast"],
            saturation=row["saturation"],
            warmth=row["warmth"],
            color_histogram=MetadataRepository._parse_histogram(row["color_histogram"]),
        )

    @staticmethod
    def _parse_histogram(raw_histogram: str | None) -> tuple[float, ...] | None:
        if not raw_histogram:
            return None

        try:
            values = json.loads(raw_histogram)
        except json.JSONDecodeError as exc:
            raise ValueError("Stored color_histogram is not valid JSON") from exc

        if not isinstance(values, list):
            raise ValueError("Stored color_histogram must decode to a list")

        return tuple(float(value) for value in values)

    @staticmethod
    def _metadata_to_style_features(
        metadata: ImageMetadata,
    ) -> dict[str, float | list[float]]:
        if not metadata.has_style_features:
            raise ValueError(f"Style features are incomplete for image_id={metadata.image_id}")

        assert metadata.brightness is not None
        assert metadata.contrast is not None
        assert metadata.saturation is not None
        assert metadata.warmth is not None
        assert metadata.color_histogram is not None

        return {
            "brightness": float(metadata.brightness),
            "contrast": float(metadata.contrast),
            "saturation": float(metadata.saturation),
            "warmth": float(metadata.warmth),
            "color_histogram": [float(value) for value in metadata.color_histogram],
        }
