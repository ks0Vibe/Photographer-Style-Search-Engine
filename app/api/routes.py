from __future__ import annotations

import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.api.dependencies import SearchApplication, get_embedding_dim, get_search_application
from app.api.schemas import (
    HealthResponse,
    ImageMetadataResponse,
    ImageSearchRequest,
    ImageSearchResponse,
    SearchResponse,
    SearchResult,
    StatsResponse,
    TextSearchRequest,
)
from app.search import ObjectAwareReranker
from app.search.qdrant_config import PROJECT_ROOT


router = APIRouter()
QUERY_UPLOAD_DIR = PROJECT_ROOT / "data" / "query_uploads"
SUPPORTED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_UPLOAD_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _style_filter_kwargs(request: TextSearchRequest | ImageSearchRequest) -> dict[str, float | None]:
    return {
        "min_brightness": request.min_brightness,
        "max_brightness": request.max_brightness,
        "min_contrast": request.min_contrast,
        "max_contrast": request.max_contrast,
        "min_saturation": request.min_saturation,
        "max_saturation": request.max_saturation,
        "min_warmth": request.min_warmth,
        "max_warmth": request.max_warmth,
    }


def _enrich_results(results: list[dict[str, Any]], app_state: SearchApplication) -> list[SearchResult]:
    metadata_by_id = app_state.metadata.get_many([str(row.get("image_id", "")) for row in results])
    output: list[SearchResult] = []
    for rank, result in enumerate(results, start=1):
        image_id = str(result.get("image_id", ""))
        metadata = metadata_by_id.get(image_id, {})
        output.append(
            SearchResult(
                rank=rank,
                image_id=image_id,
                score=float(result.get("score", result.get("final_score", 0.0)) or 0.0),
                file_path=str(result.get("file_path") or metadata.get("file_path") or ""),
                photo_url=metadata.get("photo_url"),
                ai_description=metadata.get("ai_description"),
                photographer_username=metadata.get("photographer_username"),
                keywords=list(result.get("keywords") or metadata.get("keywords") or []),
                detected_objects=list(result.get("detected_objects") or metadata.get("detected_objects") or []),
                brightness=_optional_float(result.get("brightness", metadata.get("brightness"))),
                contrast=_optional_float(result.get("contrast", metadata.get("contrast"))),
                saturation=_optional_float(result.get("saturation", metadata.get("saturation"))),
                warmth=_optional_float(result.get("warmth", metadata.get("warmth"))),
            )
        )
    return output


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/stats", response_model=StatsResponse)
def stats(app_state: SearchApplication = Depends(get_search_application)) -> StatsResponse:
    try:
        metadata_stats = app_state.metadata.stats()
        qdrant_points = app_state.qdrant_points()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return StatsResponse(
        sqlite_image_rows=metadata_stats["sqlite_image_rows"],
        qdrant_collection=app_state.settings.collection_name,
        qdrant_points=qdrant_points,
        embedding_dim=get_embedding_dim(),
        images_with_detected_objects=metadata_stats["images_with_detected_objects"],
        object_coverage=round(float(metadata_stats["object_coverage"]), 4),
    )


@router.post("/search/text", response_model=SearchResponse)
def search_text(
    request: TextSearchRequest,
    app_state: SearchApplication = Depends(get_search_application),
) -> SearchResponse:
    started = time.perf_counter()
    try:
        if request.object_rerank:
            candidates = app_state.service.search_by_text(
                text=request.query,
                top_k=request.candidate_pool_size,
                keyword_filter=request.keyword,
                rerank=False,
                candidate_pool_size=request.candidate_pool_size,
                **_style_filter_kwargs(request),
            )
            results = ObjectAwareReranker.object_heavy().rerank(
                candidates,
                requested_object=request.object,
                requested_keyword=request.keyword or request.object,
            )[: request.top_k]
            mode = "object_rerank"
        else:
            results = app_state.service.search_by_text(
                text=request.query,
                top_k=request.top_k,
                keyword_filter=request.keyword,
                object_filter=request.object,
                rerank=request.rerank,
                candidate_pool_size=request.candidate_pool_size,
                **_style_filter_kwargs(request),
            )
            if request.object:
                mode = "object_filter"
            elif request.keyword:
                mode = "keyword_filter"
            elif request.rerank:
                mode = "style_rerank"
            else:
                mode = "semantic"
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - started) * 1000.0
    return SearchResponse(
        query=request.query,
        mode=mode,
        top_k=request.top_k,
        latency_ms=latency_ms,
        results=_enrich_results(results, app_state),
    )


@router.post("/search/image", response_model=ImageSearchResponse)
def search_image(
    request: ImageSearchRequest,
    app_state: SearchApplication = Depends(get_search_application),
) -> ImageSearchResponse:
    if bool(request.image_id) == bool(request.image_path):
        raise HTTPException(status_code=400, detail="Provide exactly one of image_id or image_path")

    if request.image_id:
        image_path = app_state.metadata.resolve_image_path(request.image_id)
        query_type = "dataset_image_id"
        query_image_id = request.image_id
        query_image_path = str(image_path.relative_to(PROJECT_ROOT)) if image_path else None
    else:
        image_path = _resolve_user_image_path(str(request.image_path))
        query_type = "local_image_path"
        query_image_id = None
        query_image_path = str(image_path) if image_path else str(request.image_path)

    if image_path is None:
        status_code = 404 if request.image_id else 400
        raise HTTPException(status_code=status_code, detail="Query image not found")

    started = time.perf_counter()
    try:
        results = app_state.service.search_by_image(
            image_path=image_path,
            top_k=request.top_k,
            keyword_filter=request.keyword,
            object_filter=request.object,
            rerank=request.rerank,
            candidate_pool_size=request.candidate_pool_size,
            **_style_filter_kwargs(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - started) * 1000.0
    return ImageSearchResponse(
        query_type=query_type,
        query_image_id=query_image_id,
        query_image_path=query_image_path,
        mode="image_rerank" if request.rerank else "image_semantic",
        top_k=request.top_k,
        latency_ms=latency_ms,
        results=_enrich_results(results, app_state),
    )


@router.post("/search/image/upload", response_model=ImageSearchResponse)
async def search_uploaded_image(
    file: UploadFile = File(...),
    top_k: int = Form(10),
    candidate_pool_size: int = Form(100),
    rerank: bool = Form(False),
    app_state: SearchApplication = Depends(get_search_application),
) -> ImageSearchResponse:
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    if candidate_pool_size < 1 or candidate_pool_size > 500:
        raise HTTPException(status_code=400, detail="candidate_pool_size must be between 1 and 500")

    saved_path = await _save_uploaded_query_image(file)
    request = ImageSearchRequest(
        image_path=str(saved_path),
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        rerank=rerank,
    )

    started = time.perf_counter()
    try:
        results = app_state.service.search_by_image(
            image_path=saved_path,
            top_k=request.top_k,
            rerank=request.rerank,
            candidate_pool_size=request.candidate_pool_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - started) * 1000.0
    return ImageSearchResponse(
        query_type="uploaded_image",
        query_image_path=str(saved_path.relative_to(PROJECT_ROOT)),
        mode="image_rerank" if request.rerank else "image_semantic",
        top_k=request.top_k,
        latency_ms=latency_ms,
        results=_enrich_results(results, app_state),
    )


@router.get("/images/{image_id}", response_model=ImageMetadataResponse)
def image_metadata(
    image_id: str,
    app_state: SearchApplication = Depends(get_search_application),
) -> ImageMetadataResponse:
    metadata = app_state.metadata.get_image(image_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return ImageMetadataResponse(**metadata)


@router.get("/image-file/{image_id}")
def image_file(
    image_id: str,
    app_state: SearchApplication = Depends(get_search_application),
) -> FileResponse:
    image_path = app_state.metadata.resolve_image_path(image_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(image_path)


def _resolve_user_image_path(raw_path: str) -> Path | None:
    if not raw_path.strip():
        return None
    path = Path(raw_path)
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = resolved.resolve()
    if path.parts and any(part == ".." for part in path.parts):
        return None
    if resolved.suffix.lower() not in SUPPORTED_UPLOAD_EXTENSIONS:
        return None
    return resolved if resolved.exists() and resolved.is_file() else None


async def _save_uploaded_query_image(file: UploadFile) -> Path:
    original_name = file.filename or ""
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image extension")
    if file.content_type not in SUPPORTED_UPLOAD_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported image content type")

    QUERY_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = QUERY_UPLOAD_DIR / f"query_{uuid.uuid4().hex}{suffix}"
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        with Image.open(BytesIO(contents)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc
    target_path.write_bytes(contents)
    return target_path
