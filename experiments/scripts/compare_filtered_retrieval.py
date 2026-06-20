import csv
import statistics
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder
from app.search import QdrantRetrievalService, QdrantStore, load_keywords_by_image_id
from experiments.paths import FILTERED_RETRIEVAL_DIR, ensure_experiment_directories
from scripts.qdrant_common import COLLECTION_NAME, IMAGE_IDS_PATH, KEYWORDS_PATH, QDRANT_PATH


CSV_OUTPUT_PATH = FILTERED_RETRIEVAL_DIR / "filtered_retrieval_results.csv"
MD_OUTPUT_PATH = FILTERED_RETRIEVAL_DIR / "filtered_retrieval_results.md"
REPORT_PATH = FILTERED_RETRIEVAL_DIR / "report.md"
TOP_K = 10
TEXT_QUERIES = [
    "warm cinematic landscape",
    "dark moody forest",
    "street photography",
    "portrait",
    "beach sunset",
]
PREFERRED_KEYWORDS = {
    "warm cinematic landscape": ("nature", "water"),
    "dark moody forest": ("nature", "water"),
    "street photography": ("person", "building"),
    "portrait": ("person",),
    "beach sunset": ("water", "nature"),
}
STYLE_FILTERS = {
    "warm cinematic landscape": {"min_warmth": 0.55},
    "dark moody forest": {"max_brightness": 0.40},
    "street photography": {"max_brightness": 0.50},
    "portrait": {"min_warmth": 0.45},
    "beach sunset": {"min_warmth": 0.60, "min_saturation": 0.45},
}


def create_qdrant_service() -> QdrantRetrievalService:
    return QdrantRetrievalService(
        clip_encoder=CLIPEncoder(),
        qdrant_store=QdrantStore(
            collection_name=COLLECTION_NAME,
            qdrant_path=QDRANT_PATH,
        ),
    )


def load_available_keywords() -> set[str]:
    image_ids = {str(image_id) for image_id in np.load(IMAGE_IDS_PATH, allow_pickle=False).tolist()}
    if not KEYWORDS_PATH.exists():
        return set()

    keywords_by_image_id = load_keywords_by_image_id(KEYWORDS_PATH)
    available_keywords: set[str] = set()
    for image_id, keywords in keywords_by_image_id.items():
        if image_id not in image_ids:
            continue
        available_keywords.update(keywords)
    return available_keywords


def summarize_results(results: list[dict]) -> dict[str, object]:
    return {
        "result_count": len(results),
        "top1_image_id": str(results[0]["image_id"]) if results else "",
        "top1_score": float(results[0]["score"]) if results else 0.0,
        "avg_score": statistics.fmean(float(result["score"]) for result in results) if results else 0.0,
        "avg_brightness": statistics.fmean(
            float(result["brightness"]) for result in results if result["brightness"] is not None
        ) if any(result["brightness"] is not None for result in results) else 0.0,
        "avg_warmth": statistics.fmean(
            float(result["warmth"]) for result in results if result["warmth"] is not None
        ) if any(result["warmth"] is not None for result in results) else 0.0,
    }


def select_keyword_filter(query: str, available_keywords: set[str]) -> str | None:
    for keyword in PREFERRED_KEYWORDS.get(query, ()):
        if keyword in available_keywords:
            return keyword
    return None


def run_mode(
    service: QdrantRetrievalService,
    query: str,
    *,
    keyword_filter: str | None = None,
    rerank: bool = False,
    style_filters: dict[str, float] | None = None,
) -> list[dict]:
    kwargs = {
        "text": query,
        "top_k": TOP_K,
        "keyword_filter": keyword_filter,
        "object_filter": None,
        "rerank": rerank,
        "candidate_pool_size": 100,
    }
    if style_filters:
        kwargs.update(style_filters)
    return service.search_by_text(**kwargs)


def write_csv(rows: list[dict[str, object]]) -> None:
    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "mode",
                "filter_value",
                "result_count",
                "top1_image_id",
                "top1_score",
                "avg_score",
                "avg_brightness",
                "avg_warmth",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Filtered Retrieval Comparison",
        "",
        "This stage exercises the Qdrant payload features that are not available in the FAISS baseline: keyword filters, style-range filters, and optional reranking on the filtered candidate set.",
        "",
        "Compared modes:",
        "",
        "- Qdrant semantic search only",
        "- Qdrant semantic search + keyword filter",
        "- Qdrant semantic search + style filter",
        "- Qdrant semantic search + reranking",
        "",
        "| Query | Mode | Filter | Count | Top-1 | Top-1 Score | Avg Score | Avg Brightness | Avg Warmth |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| {query} | {mode} | {filter_value} | {result_count} | {top1_image_id} | {top1_score:.4f} | {avg_score:.4f} | {avg_brightness:.4f} | {avg_warmth:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python experiments/scripts/compare_filtered_retrieval.py",
            "```",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ensure_experiment_directories()
    available_keywords = load_available_keywords()
    service = create_qdrant_service()
    rows: list[dict[str, object]] = []
    try:
        for query in TEXT_QUERIES:
            semantic_results = run_mode(service, query)
            rows.append(
                {
                    "query": query,
                    "mode": "semantic_only",
                    "filter_value": "",
                    **summarize_results(semantic_results),
                }
            )

            keyword_filter = select_keyword_filter(query, available_keywords)
            if keyword_filter is not None:
                keyword_results = run_mode(service, query, keyword_filter=keyword_filter)
                rows.append(
                    {
                        "query": query,
                        "mode": "keyword_filter",
                        "filter_value": keyword_filter,
                        **summarize_results(keyword_results),
                    }
                )

            style_filter_kwargs = STYLE_FILTERS.get(query, {})
            style_results = run_mode(service, query, style_filters=style_filter_kwargs)
            rows.append(
                {
                    "query": query,
                    "mode": "style_filter",
                    "filter_value": ",".join(f"{key}={value}" for key, value in style_filter_kwargs.items()),
                    **summarize_results(style_results),
                }
            )

            reranked_results = run_mode(service, query, rerank=True)
            rows.append(
                {
                    "query": query,
                    "mode": "rerank",
                    "filter_value": "style_rerank",
                    **summarize_results(reranked_results),
                }
            )
    finally:
        service.close()

    write_csv(rows)
    markdown = build_markdown(rows)
    MD_OUTPUT_PATH.write_text(markdown, encoding="utf-8")
    REPORT_PATH.write_text(markdown, encoding="utf-8")

    print(f"Saved CSV: {CSV_OUTPUT_PATH}")
    print(f"Saved report: {MD_OUTPUT_PATH}")
    print(f"Saved stage report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
