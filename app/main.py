from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.api.dependencies import clear_search_application_cache, get_search_application, search_application_cache_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        if search_application_cache_info().currsize:
            get_search_application().close()
            clear_search_application_cache()


app = FastAPI(
    title="Photographer Style Search Engine",
    version="0.1.0",
    description="Local demo API for CLIP/Qdrant image retrieval with metadata, style, and YOLO object signals.",
    lifespan=lifespan,
)
app.include_router(router)
