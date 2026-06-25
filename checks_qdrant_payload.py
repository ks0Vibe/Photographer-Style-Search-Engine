from collections import Counter
from qdrant_client import QdrantClient

QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "photos"

client = QdrantClient(path=QDRANT_PATH)

try:
    info = client.get_collection(COLLECTION_NAME)

    print("Collection:", COLLECTION_NAME)
    print("Points count:", info.points_count)
    print()

    offset = None
    total_seen = 0

    keyword_non_empty = 0
    object_non_empty = 0

    keyword_counter = Counter()
    object_counter = Counter()

    sample_payloads = []

    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            total_seen += 1
            payload = point.payload or {}

            if len(sample_payloads) < 3:
                sample_payloads.append(payload)

            keywords = payload.get("keywords", [])
            detected_objects = payload.get("detected_objects", [])

            if isinstance(keywords, str):
                keywords = [keywords]

            if isinstance(detected_objects, str):
                detected_objects = [detected_objects]

            keywords = [
                str(x).lower().strip()
                for x in keywords
                if str(x).strip()
            ]

            detected_objects = [
                str(x).lower().strip()
                for x in detected_objects
                if str(x).strip()
            ]

            if keywords:
                keyword_non_empty += 1

            if detected_objects:
                object_non_empty += 1

            keyword_counter.update(keywords)
            object_counter.update(detected_objects)

        if offset is None:
            break

    print("Total scanned points:", total_seen)
    print()

    print("Payload samples:")
    for payload in sample_payloads:
        print(payload)
        print("---")

    print()
    print("Images with non-empty keywords:", keyword_non_empty)
    print("Keyword coverage:", keyword_non_empty / total_seen if total_seen else 0)

    print()
    print("Images with non-empty detected_objects:", object_non_empty)
    print("Object coverage:", object_non_empty / total_seen if total_seen else 0)

    print()
    print("Top 50 payload keywords:")
    for keyword, count in keyword_counter.most_common(50):
        print(f"{keyword}: {count}")

    print()
    print("Top 50 detected objects:")
    for obj, count in object_counter.most_common(50):
        print(f"{obj}: {count}")

finally:
    client.close()