from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.search import ObjectAwareReranker
from scripts.qdrant_common import OUTPUT_DIR, create_qdrant_service
from scripts.visualize_text_search import build_result_grid, sanitize_query_for_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Qdrant object-aware text retrieval.")
    parser.add_argument("--query", required=True, help="Text query.")
    parser.add_argument("--object", dest="object_filter", required=True, help="Requested object label.")
    parser.add_argument("--keyword", default=None, help="Optional keyword filter/score label.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to visualize.")
    parser.add_argument("--candidate-pool-size", type=int, default=100, help="Candidate pool before object-aware rerank.")
    parser.add_argument("--object-rerank", action="store_true", help="Apply object-aware reranking instead of strict object filtering only.")
    parser.add_argument("--output", default=None, help="Optional output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = create_qdrant_service()
    try:
        if args.object_rerank:
            results = service.search_by_text(
                text=args.query,
                top_k=args.candidate_pool_size,
                keyword_filter=args.keyword,
                rerank=False,
            )
            results = ObjectAwareReranker.object_heavy().rerank(
                results,
                requested_object=args.object_filter,
                requested_keyword=args.keyword or args.object_filter,
            )[: args.top_k]
        else:
            results = service.search_by_text(
                text=args.query,
                top_k=args.top_k,
                keyword_filter=args.keyword,
                object_filter=args.object_filter,
                rerank=False,
            )
    finally:
        service.close()

    if not results:
        raise RuntimeError("Object search returned no results")

    grid = build_result_grid(query=f"{args.query} / object={args.object_filter}", results=results)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        suffix = "object_rerank" if args.object_rerank else "object_filter"
        output_path = OUTPUT_DIR / (
            f"qdrant_object_{sanitize_query_for_filename(args.query)}_"
            f"{sanitize_query_for_filename(args.object_filter)}_{suffix}.jpg"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
