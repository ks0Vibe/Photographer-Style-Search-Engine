from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.search import ObjectAwareReranker
from scripts.qdrant_common import create_qdrant_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qdrant object-aware text retrieval.")
    parser.add_argument("--query", required=True, help="Text query.")
    parser.add_argument("--object", dest="object_filter", required=True, help="Requested object label.")
    parser.add_argument("--keyword", default=None, help="Optional keyword filter/score label.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to print.")
    parser.add_argument("--candidate-pool-size", type=int, default=100, help="Candidate pool before object-aware rerank.")
    parser.add_argument("--object-rerank", action="store_true", help="Apply object-aware reranking instead of strict object filtering only.")
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
            reranker = ObjectAwareReranker.object_heavy()
            results = reranker.rerank(
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

    print(f"Query: {args.query}")
    print(f"Object: {args.object_filter}")
    print()
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. image_id={result['image_id']} score={float(result['score']):.4f} "
            f"objects={result.get('detected_objects', [])} keywords={result.get('keywords', [])[:8]} "
            f"path={result['file_path']}"
        )


if __name__ == "__main__":
    main()
