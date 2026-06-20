import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qdrant_common import add_filter_args, create_qdrant_service, extract_filter_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Qdrant text-to-image retrieval smoke test.")
    parser.add_argument("--query", type=str, required=True, help="Text query to search for.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to display.")
    add_filter_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = create_qdrant_service()
    try:
        results = service.search_by_text(
            text=args.query,
            top_k=args.top_k,
            **extract_filter_kwargs(args),
        )
    finally:
        service.close()

    print("Query:")
    print(args.query)
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
