from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "photos"

keywords_to_check = [
    "person",
    "human",
    "people",
    "street",
    "road",
    "car",
    "dog",
    "cat",
    "food",
    "flower",
    "building",
    "forest",
    "beach",
    "night",
    "portrait",
    "architecture",
    "mountain",
    "water",
    "sky",
    "sunset",
    "nature",
]

client = QdrantClient(path=QDRANT_PATH)

try:
    for keyword in keywords_to_check:
        qfilter = Filter(
            must=[
                FieldCondition(
                    key="keywords",
                    match=MatchValue(value=keyword),
                )
            ]
        )

        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qfilter,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

        print("=" * 80)
        print(f"KEYWORD: {keyword}")
        print(f"Returned sample points: {len(points)}")

        for point in points[:3]:
            payload = point.payload or {}
            print("image_id:", payload.get("image_id"))
            print("ai_description:", payload.get("ai_description"))
            print("keywords:", payload.get("keywords"))
            print()

finally:
    client.close()