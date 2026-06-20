import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paths import STYLE_RERANKING_VISUALIZATIONS_DIR
from scripts.visualize_search_results import (
    build_comparison_grid,
    create_retrieval_service,
    filter_query_from_results,
    resolve_external_image_path,
    resolve_query_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a side-by-side comparison of CLIP-only vs style-reranked results."
    )

    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--image-id", type=str, help="Image ID from the database.")
    query_group.add_argument("--image-path", type=str, help="External image path.")

    parser.add_argument("--top-k", type=int, default=10, help="Number of results per row.")
    parser.add_argument("--output", type=str, default=None, help="Output image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = create_retrieval_service()

    if args.image_id:
        query_image_path = resolve_query_image_path(service, args.image_id)
        output_name = f"compare_{args.image_id}.jpg"
    else:
        query_image_path = resolve_external_image_path(args.image_path)
        output_name = f"compare_{query_image_path.stem}.jpg"

    if not query_image_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_image_path}")
    if args.top_k <= 0:
        raise ValueError(f"top_k must be greater than 0, got {args.top_k}")

    requested_top_k = args.top_k + 1
    clip_only_results = service.search_by_image(
        image_path=query_image_path,
        top_k=requested_top_k,
        rerank_enabled=False,
    )
    reranked_results = service.search_by_image(
        image_path=query_image_path,
        top_k=requested_top_k,
        rerank_enabled=True,
    )

    clip_only_results = filter_query_from_results(query_image_path, clip_only_results, args.top_k)
    reranked_results = filter_query_from_results(query_image_path, reranked_results, args.top_k)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = STYLE_RERANKING_VISUALIZATIONS_DIR / output_name

    build_comparison_grid(
        query_image_path=query_image_path,
        clip_only_results=clip_only_results,
        reranked_results=reranked_results,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
