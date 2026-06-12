import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.clip_encoder import CLIPEncoder
from app.search import FaissIndex, MetadataRepository, RetrievalService, VectorStore


EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "clip_embeddings.npy"
IMAGE_IDS_PATH = PROJECT_ROOT / "data" / "embeddings" / "image_ids.npy"
INDEX_PATH = PROJECT_ROOT / "data" / "indexes" / "flat.index"
DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "visualizations"


def create_retrieval_service() -> RetrievalService:
    clip_encoder = CLIPEncoder()

    vector_store = VectorStore(
        embeddings_path=EMBEDDINGS_PATH,
        image_ids_path=IMAGE_IDS_PATH,
    )

    faiss_index = FaissIndex(
        index_path=INDEX_PATH,
        dimension=512,
    )
    faiss_index.load()

    metadata_repository = MetadataRepository(
        database_path=DATABASE_PATH,
    )

    return RetrievalService(
        clip_encoder=clip_encoder,
        vector_store=vector_store,
        faiss_index=faiss_index,
        metadata_repository=metadata_repository,
    )


def get_metadata_by_id(
    metadata_repository: MetadataRepository,
    image_id: str,
):
    metadata = metadata_repository.get_by_id(image_id)

    if metadata is None:
        raise ValueError(f"Image ID not found in database: {image_id}")

    return metadata


def resolve_query_image_path(
    service: RetrievalService,
    image_id: str,
) -> Path:
    metadata = get_metadata_by_id(service.metadata_repository, image_id)
    file_path = metadata.file_path

    path = Path(file_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def resolve_external_image_path(image_path: str) -> Path:
    path = Path(image_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_image_to_canvas(
    path: Path,
    size: tuple[int, int],
) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size)

    canvas = Image.new("RGB", size, "white")

    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


def get_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        subtitle_font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()

    return title_font, subtitle_font


def draw_card_label(
    image: Image.Image,
    title: str,
    subtitle_lines: list[str] | None = None,
) -> Image.Image:
    lines = subtitle_lines or []
    label_height = 48 + (18 * len(lines))

    output = Image.new(
        "RGB",
        (image.width, image.height + label_height),
        "white",
    )

    output.paste(image, (0, label_height))

    draw = ImageDraw.Draw(output)
    title_font, subtitle_font = get_fonts()

    draw.text((8, 8), title, fill="black", font=title_font)

    y = 34
    for line in lines:
        draw.text((8, y), line, fill="gray", font=subtitle_font)
        y += 18

    return output


def resolve_result_path(result: dict[str, object]) -> Path:
    file_path = result.get("file_path")

    if not file_path:
        raise ValueError(f"Search result has no file_path: {result}")

    path = Path(file_path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def filter_query_from_results(
    query_image_path: Path,
    results: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    filtered_results: list[dict[str, object]] = []
    resolved_query_path = query_image_path.resolve()

    for result in results:
        result_path = resolve_result_path(result).resolve()
        if result_path == resolved_query_path:
            continue
        filtered_results.append(result)
        if len(filtered_results) >= top_k:
            break

    return filtered_results


def build_grid(
    query_image_path: Path,
    results: list[dict[str, object]],
    output_path: Path,
    section_title: str | None = None,
    image_size: tuple[int, int] = (220, 220),
    columns: int = 5,
) -> Image.Image:
    cards: list[Image.Image] = []

    query_image = load_image_to_canvas(query_image_path, image_size)
    query_card = draw_card_label(
        query_image,
        title="QUERY",
        subtitle_lines=[query_image_path.name],
    )
    cards.append(query_card)

    if section_title:
        section_card = Image.new("RGB", query_card.size, "white")
        draw = ImageDraw.Draw(section_card)
        title_font, _ = get_fonts()
        draw.text((8, 8), section_title, fill="black", font=title_font)
        cards.append(section_card)

    for rank, result in enumerate(results, start=1):
        result_path = resolve_result_path(result)

        result_image = load_image_to_canvas(result_path, image_size)

        image_id = str(result.get("image_id", "unknown"))
        score = float(result.get("score", 0.0))
        semantic_score = float(result.get("semantic_score", score))
        style_score = result.get("style_score")
        final_score = float(result.get("final_score", score))

        subtitle_lines = [image_id, f"sem={semantic_score:.4f}"]
        if style_score is not None:
            subtitle_lines.append(f"style={float(style_score):.4f}")
            subtitle_lines.append(f"final={final_score:.4f}")

        result_card = draw_card_label(
            result_image,
            title=f"#{rank} score={score:.4f}",
            subtitle_lines=subtitle_lines,
        )

        cards.append(result_card)

    if not cards:
        raise RuntimeError("No cards to visualize")

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

    grid = Image.new(
        "RGB",
        (columns * card_width, rows * card_height),
        "white",
    )

    for index, card in enumerate(normalized_cards):
        x = (index % columns) * card_width
        y = (index // columns) * card_height
        grid.paste(card, (x, y))

    return grid


def save_grid(grid: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Saved visualization to: {output_path}")


def build_comparison_grid(
    query_image_path: Path,
    clip_only_results: list[dict[str, object]],
    reranked_results: list[dict[str, object]],
    output_path: Path,
) -> None:
    clip_grid = build_grid(
        query_image_path=query_image_path,
        results=clip_only_results,
        output_path=output_path,
        section_title="CLIP ONLY",
    )
    reranked_grid = build_grid(
        query_image_path=query_image_path,
        results=reranked_results,
        output_path=output_path,
        section_title="CLIP + STYLE RERANK",
    )

    combined = Image.new(
        "RGB",
        (max(clip_grid.width, reranked_grid.width), clip_grid.height + reranked_grid.height),
        "white",
    )
    combined.paste(clip_grid, (0, 0))
    combined.paste(reranked_grid, (0, clip_grid.height))

    save_grid(combined, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize image-to-image search results as a grid."
    )

    query_group = parser.add_mutually_exclusive_group(required=True)

    query_group.add_argument(
        "--image-id",
        type=str,
        help="Image ID from the database to use as the query image.",
    )

    query_group.add_argument(
        "--image-path",
        type=str,
        help="Path to an external image to use as the query image.",
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
        help="Output image path.",
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Generate a visual comparison between CLIP-only and style-reranked results.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    service = create_retrieval_service()

    if args.image_id:
        query_image_path = resolve_query_image_path(
            service=service,
            image_id=args.image_id,
        )
        output_name = f"search_{args.image_id}.jpg"
    else:
        query_image_path = resolve_external_image_path(args.image_path)
        output_name = f"search_{query_image_path.stem}.jpg"

    if not query_image_path.exists():
        raise FileNotFoundError(f"Query image not found: {query_image_path}")

    print(f"Query image: {query_image_path}")

    requested_top_k = args.top_k + 1

    results = service.search_by_image(
        image_path=query_image_path,
        top_k=requested_top_k,
        rerank_enabled=args.rerank,
    )
    results = filter_query_from_results(query_image_path, results, args.top_k)

    if not results:
        raise RuntimeError("Search returned no results")

    print("\nResults:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.get('image_id')} "
            f"score={float(result.get('score', 0.0)):.4f} "
            f"path={result.get('file_path')}"
        )

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        safe_output_name = output_name.replace("/", "_").replace("\\", "_")
        output_path = OUTPUT_DIR / safe_output_name

    if args.rerank:
        clip_only_results = service.search_by_image(
            image_path=query_image_path,
            top_k=requested_top_k,
            rerank_enabled=False,
        )
        clip_only_results = filter_query_from_results(query_image_path, clip_only_results, args.top_k)
        build_comparison_grid(
            query_image_path=query_image_path,
            clip_only_results=clip_only_results,
            reranked_results=results,
            output_path=output_path,
        )
    else:
        grid = build_grid(
            query_image_path=query_image_path,
            results=results,
            output_path=output_path,
        )
        save_grid(grid, output_path)


if __name__ == "__main__":
    main()
