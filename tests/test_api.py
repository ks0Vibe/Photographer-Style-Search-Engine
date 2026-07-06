from __future__ import annotations

from pathlib import Path
from io import BytesIO
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def local_artifacts_available() -> bool:
    return (
        (PROJECT_ROOT / "data" / "metadata.sqlite").exists()
        and (PROJECT_ROOT / "data" / "qdrant").exists()
        and (PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy").exists()
    )


client = TestClient(app)


def first_image_id() -> str:
    with sqlite3.connect(PROJECT_ROOT / "data" / "metadata.sqlite") as conn:
        row = conn.execute("SELECT image_id FROM images ORDER BY image_id LIMIT 1").fetchone()
    assert row is not None
    return str(row[0])


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(not local_artifacts_available(), reason="Local dataset/Qdrant artifacts are not available")
def test_stats() -> None:
    response = client.get("/stats")
    assert response.status_code == 200
    payload = response.json()
    assert payload["sqlite_image_rows"] > 0
    assert payload["qdrant_collection"] == "photos"
    assert payload["qdrant_points"] > 0
    assert payload["embedding_dim"] == 512


@pytest.mark.skipif(not local_artifacts_available(), reason="Local dataset/Qdrant artifacts are not available")
def test_text_search_and_image_metadata() -> None:
    response = client.post(
        "/search/text",
        json={
            "query": "dog on beach",
            "top_k": 3,
            "candidate_pool_size": 50,
            "object": "dog",
            "object_rerank": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]

    image_id = payload["results"][0]["image_id"]
    metadata_response = client.get(f"/images/{image_id}")
    assert metadata_response.status_code == 200
    assert metadata_response.json()["image_id"] == image_id


@pytest.mark.skipif(not local_artifacts_available(), reason="Local dataset/Qdrant artifacts are not available")
def test_image_search_by_existing_image_id() -> None:
    response = client.post(
        "/search/image",
        json={
            "image_id": first_image_id(),
            "top_k": 1,
            "candidate_pool_size": 5,
            "rerank": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["query_type"] == "dataset_image_id"
    assert payload["query_image_id"]
    assert payload["mode"] == "image_semantic"
    assert "results" in payload


@pytest.mark.skipif(not local_artifacts_available(), reason="Local dataset/Qdrant artifacts are not available")
def test_image_search_upload_smoke() -> None:
    image = Image.new("RGB", (16, 16), color=(120, 80, 40))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/search/image/upload",
        data={
            "top_k": "1",
            "candidate_pool_size": "5",
            "rerank": "false",
        },
        files={"file": ("query.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["query_type"] == "uploaded_image"
    assert payload["query_image_path"].startswith("data")
    assert "results" in payload
