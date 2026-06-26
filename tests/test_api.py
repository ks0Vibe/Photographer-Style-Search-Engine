from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def local_artifacts_available() -> bool:
    return (
        (PROJECT_ROOT / "data" / "metadata.sqlite").exists()
        and (PROJECT_ROOT / "data" / "qdrant").exists()
        and (PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy").exists()
    )


client = TestClient(app)


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
