import argparse
import logging
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.visualize_search_results import (
    OUTPUT_DIR,
    draw_card_label,
    get_fonts,
    load_image_to_canvas,
    resolve_result_path,
    save_grid,
)
from app.search import RetrievalService
from scripts.visualize_search_results import create_retrieval_service


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize text-to-image search results as a grid."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query to run through the CLIP text-to-image pipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of search results to visualize.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output image path.",
    )
    return parser.parse_args()


def sanitize_query_for_filename(query: str) -> str:
    collapsed = "_".join(query.strip().lower().split())
    sanitized = re.sub(r"[^a-z0-9_]+", "", collapsed)
    return sanitized or "query"


def build_query_header(
    query: str,
    width: int,
    padding: int = 12,
) -> Image.Image:
    title_font, subtitle_font = get_fonts()
    header_height = 96
    header = Image.new("RGB", (width, header_height), "white")
    draw = ImageDraw.Draw(header)

    draw.text((padding, 10), "TEXT QUERY", fill="black", font=title_font)
    draw.text((padding, 40), query, fill="gray", font=subtitle_font)
    return header


def build_result_grid(
    query: str,
    results: list[dict[str, str | float]],
    image_size: tuple[int, int] = (220, 220),
    columns: int = 5,
) -> Image.Image:
    cards: list[Image.Image] = []

    for rank, result in enumerate(results, start=1):
        result_path = resolve_result_path(result)
        result_image = load_image_to_canvas(result_path, image_size)

        image_id = str(result.get("image_id", "unknown"))
        score = float(result.get("score", 0.0))

        card = draw_card_label(
            result_image,
            title=f"#{rank} score={score:.4f}",
            subtitle_lines=[image_id],
        )
        cards.append(card)

    if not cards:
        raise RuntimeError("Search returned no visualizable results")

    card_width = max(card.width for card in cards)
    card_height = max(card.height for card in cards)

    normalized_cards: list[Image.Image] = []
    for card in cards:
        if card.width == card_width and card.height == card_height:
            normalized_cards.append(card)
            continue

        padded = Image.new("RGB", (card_width, card_height), "white")
        padded.paste(card, (0, 0))
        normalized_cards.append(padded)

    rows = (len(normalized_cards) + columns - 1) // columns
    grid_width = columns * card_width
    grid_height = rows * card_height

    results_grid = Image.new("RGB", (grid_width, grid_height), "white")
    for index, card in enumerate(normalized_cards):
        x = (index % columns) * card_width
        y = (index // columns) * card_height
        results_grid.paste(card, (x, y))

    header = build_query_header(query=query, width=grid_width)
    output = Image.new("RGB", (grid_width, header.height + results_grid.height), "white")
    output.paste(header, (0, 0))
    output.paste(results_grid, (0, header.height))
    return output


def resolve_output_path(query: str, output: str | None) -> Path:
    if output:
        output_path = Path(output)
        return output_path if output_path.is_absolute() else PROJECT_ROOT / output_path

    file_name = f"text_{sanitize_query_for_filename(query)}.jpg"
    return OUTPUT_DIR / file_name


def run_search(
    service: RetrievalService,
    query: str,
    top_k: int,
) -> list[dict[str, str | float]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than 0, got {top_k}")

    logger.info("Running text-to-image search for query=%r top_k=%s", query, top_k)
    return service.search_by_text(query, top_k=top_k)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    service = create_retrieval_service()
    results = run_search(service, query=args.query, top_k=args.top_k)
    if not results:
        raise RuntimeError("Text search returned no results")

    print(f"Query: {args.query}")
    print()
    print("Results:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.get('image_id')} "
            f"score={float(result.get('score', 0.0)):.4f} "
            f"path={result.get('file_path')}"
        )

    grid = build_result_grid(query=args.query, results=results)
    output_path = resolve_output_path(query=args.query, output=args.output)
    save_grid(grid, output_path)


if __name__ == "__main__":
    main()
