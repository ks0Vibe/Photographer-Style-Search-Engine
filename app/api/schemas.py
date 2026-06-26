from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class StatsResponse(BaseModel):
    sqlite_image_rows: int
    qdrant_collection: str
    qdrant_points: int
    embedding_dim: int | None = None
    images_with_detected_objects: int
    object_coverage: float


class StyleFilters(BaseModel):
    min_brightness: float | None = None
    max_brightness: float | None = None
    min_contrast: float | None = None
    max_contrast: float | None = None
    min_saturation: float | None = None
    max_saturation: float | None = None
    min_warmth: float | None = None
    max_warmth: float | None = None


class TextSearchRequest(StyleFilters):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_pool_size: int = Field(default=100, ge=1, le=500)
    keyword: str | None = None
    object: str | None = None
    rerank: bool = False
    object_rerank: bool = False


class ImageSearchRequest(StyleFilters):
    image_id: str | None = None
    image_path: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_pool_size: int = Field(default=100, ge=1, le=500)
    keyword: str | None = None
    object: str | None = None
    rerank: bool = True


class SearchResult(BaseModel):
    rank: int
    image_id: str
    score: float
    file_path: str
    photo_url: str | None = None
    ai_description: str | None = None
    photographer_username: str | None = None
    keywords: list[str] = Field(default_factory=list)
    detected_objects: list[str] = Field(default_factory=list)
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    warmth: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    top_k: int
    latency_ms: float
    results: list[SearchResult]


class ImageMetadataResponse(BaseModel):
    image_id: str
    file_path: str
    photo_url: str | None = None
    ai_description: str | None = None
    photographer_username: str | None = None
    keywords: list[str] = Field(default_factory=list)
    detected_objects: list[str] = Field(default_factory=list)
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    warmth: float | None = None
