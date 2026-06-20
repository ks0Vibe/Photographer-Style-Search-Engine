import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qdrant_common import (
    OUTPUT_DIR,
    add_filter_args,
    create_qdrant_service,
    extract_filter_kwargs,
    get_query_row,
    resolve_image_path,
)
from scripts.visualize_search_results import build_grid, filter_query_from_results, save_grid
from scripts.visualize_text_search import sanitize_query_for_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Qdrant image-to-image search results.")
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--image-id", type=str, default=None, help="Image ID to use as the query image.")
    query_group.add_argument("--image-path", type=str, default=None, help="External image path to use as the query image.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to visualize.")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path.")
    add_filter_args(parser)
    return parser.parse_args()


def build_default_output_name(args: argparse.Namespace, stem: str) -> str:
    parts = [f"qdrant_image_{sanitize_query_for_filename(stem)}"]
    if args.keyword:
        parts.append(f"keyword_{sanitize_query_for_filename(args.keyword)}")
    if args.object_filter:
        parts.append(f"object_{sanitize_query_for_filename(args.object_filter)}")
    if args.rerank:
        parts.append("rerank")
    return "_".join(parts) + ".jpg"


def main() -> None:
    args = parse_args()

    if args.image_path:
        query_image_path = resolve_image_path(args.image_path)
        query_image_id = None
        output_stem = Path(args.image_path).stem
    else:
        query_image_id, file_path = get_query_row(args.image_id)
        query_image_path = resolve_image_path(file_path)
        output_stem = query_image_id

    if not query_image_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_image_path}")

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
        results = filter_query_from_results(query_image_path, results, args.top_k)

    if not results:
        raise RuntimeError("Image search returned no results")

    print(f"Query image: {query_image_path}")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['image_id']} score={result['score']:.4f} path={result['file_path']}"
        )

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = OUTPUT_DIR / build_default_output_name(args, output_stem)

    grid = build_grid(
        query_image_path=query_image_path,
        results=results,
        output_path=output_path,
    )
    save_grid(grid, output_path)


if __name__ == "__main__":
    main()
