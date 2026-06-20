import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.qdrant_common import OUTPUT_DIR, add_filter_args, create_qdrant_service, extract_filter_kwargs
from scripts.visualize_text_search import build_result_grid, resolve_output_path, sanitize_query_for_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Qdrant text-to-image search results.")
    parser.add_argument("--query", type=str, required=True, help="Text query to search for.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of search results to visualize.")
    parser.add_argument("--output", type=str, default=None, help="Optional output image path.")
    add_filter_args(parser)
    return parser.parse_args()


def build_default_output_name(args: argparse.Namespace) -> str:
    parts = [f"qdrant_text_{sanitize_query_for_filename(args.query)}"]
    if args.keyword:
        parts.append(f"keyword_{sanitize_query_for_filename(args.keyword)}")
    if args.object_filter:
        parts.append(f"object_{sanitize_query_for_filename(args.object_filter)}")
    if args.rerank:
        parts.append("rerank")
    return "_".join(parts) + ".jpg"


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

    if not results:
        raise RuntimeError("Text search returned no results")

    print(f"Query: {args.query}")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['image_id']} score={result['score']:.4f} path={result['file_path']}"
        )

    grid = build_result_grid(query=args.query, results=results)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        output_path = OUTPUT_DIR / build_default_output_name(args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
