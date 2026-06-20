import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qdrant_common import (
    add_filter_args,
    create_qdrant_service,
    extract_filter_kwargs,
    get_query_row,
    resolve_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Qdrant image-to-image retrieval smoke test.")
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--image-id", type=str, default=None, help="Image ID to use as the query.")
    query_group.add_argument("--image-path", type=str, default=None, help="External image path to use as the query.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to display.")
    add_filter_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image_path:
        query_image_path = resolve_image_path(args.image_path)
        query_image_id = None
    else:
        query_image_id, file_path = get_query_row(args.image_id)
        query_image_path = resolve_image_path(file_path)

    service = create_qdrant_service()
    try:
        requested_top_k = args.top_k + 1 if query_image_id else args.top_k
        results = service.search_by_image(
            image_path=query_image_path,
            top_k=requested_top_k,
            **extract_filter_kwargs(args),
        )
    finally:
        service.close()

    if query_image_id:
        results = [result for result in results if result["image_id"] != query_image_id][: args.top_k]

    print("Query:")
    print(query_image_id or str(query_image_path))
    print()
    print("Results:")
    print()

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. image_id={result['image_id']} "
            f"score={result['score']:.4f} "
            f"file_path={result['file_path']}"
        )
        print(f"   keywords={result['keywords']}")
        print(f"   detected_objects={result['detected_objects']}")
        print(f"   brightness={result['brightness']}")
        print(f"   warmth={result['warmth']}")


if __name__ == "__main__":
    main()
