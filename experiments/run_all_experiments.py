from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.paths import (
    CLIP_BASELINE_DIR,
    CLIP_BASELINE_VISUALIZATIONS_DIR,
    FINAL_REPORT_DIR,
    FINAL_REPORT_SELECTED_VISUALIZATIONS_DIR,
    QDRANT_BACKEND_VISUALIZATIONS_DIR,
    STYLE_RERANKING_VISUALIZATIONS_DIR,
    ensure_experiment_directories,
)


BASELINE_QUERIES = [
    "warm cinematic landscape",
    "dark moody forest",
    "cold snowy mountain",
    "bright tropical beach",
    "minimal street photography",
]
RERANKING_IMAGE_IDS = [
    "9U_uCvfpptk",
    "9wTWFyInJ4Y",
    "39DcBUbYZP4",
    "A-G8q9zorGs",
]
STAGE_05_DIR = PROJECT_ROOT / "experiments" / "05_scaled_retrieval_quality"
STAGE_06_DIR = PROJECT_ROOT / "experiments" / "06_yolo_object_retrieval"
FINAL_REPORT_VISUALS = [
    STAGE_05_DIR / "visualizations" / "dog_on_beach__qdrant_keyword_primary.png",
    STAGE_05_DIR / "visualizations" / "person_in_street_photography__qdrant_keyword_primary.png",
    STAGE_05_DIR / "visualizations" / "dark_moody_forest__qdrant_rerank.png",
    STAGE_05_DIR / "visualizations" / "minimal_architecture__qdrant_semantic.png",
    STAGE_06_DIR / "visualizations" / "person__qdrant_object.png",
    STAGE_06_DIR / "visualizations" / "car_at_night__qdrant_object.png",
    STAGE_06_DIR / "visualizations" / "dog_on_beach__qdrant_object_rerank.png",
    STAGE_06_DIR / "visualizations" / "bird_in_nature__qdrant_keyword_object.png",
]


def resolve_python_executable() -> Path:
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return venv_python if venv_python.exists() else Path(sys.executable)


def run_command(description: str, args: list[str]) -> None:
    print(f"[run] {description}", flush=True)
    subprocess.run(args, check=True, cwd=PROJECT_ROOT)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_metric(path: Path, mode: str, metric: str) -> str:
    values = [
        float(row[metric])
        for row in read_csv(path)
        if row.get("mode") == mode and row.get(metric) not in {"", None}
    ]
    if not values:
        return "n/a"
    return f"{sum(values) / len(values):.4f}"


def copy_selected_visualizations() -> list[Path]:
    FINAL_REPORT_SELECTED_VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for existing_file in FINAL_REPORT_SELECTED_VISUALIZATIONS_DIR.glob("*"):
        if existing_file.is_file():
            existing_file.unlink()

    copied: list[Path] = []
    for source in FINAL_REPORT_VISUALS[:8]:
        if source.exists():
            target = FINAL_REPORT_SELECTED_VISUALIZATIONS_DIR / source.name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def format_float(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def assemble_final_report() -> None:
    selected_visuals = copy_selected_visualizations()
    stage05_payload = read_json(STAGE_05_DIR / "qdrant_payload_stats.json")
    stage06_payload = read_json(STAGE_06_DIR / "object_payload_stats.json")
    stage06_metrics_path = STAGE_06_DIR / "retrieval_metrics.csv"
    keyword_avg_relevance = aggregate_metric(stage06_metrics_path, "qdrant_keyword", "avg_relevance")
    object_avg_relevance = aggregate_metric(stage06_metrics_path, "qdrant_object", "avg_relevance")
    rerank_ndcg = aggregate_metric(stage06_metrics_path, "qdrant_object_rerank", "ndcg_at_10")

    lines = [
        "# Photographer Style Search Engine: Final Report",
        "",
        "## 1. Project Goal",
        "",
        "This project is a photographer-oriented visual search engine for retrieving images by semantic content, visual style, metadata filters, and detected objects. It starts from CLIP semantic search and progressively adds style descriptors, Qdrant payload filtering, scaled diagnostics, YOLO object evidence, object-aware reranking, and a local application layer.",
        "",
        "## 2. System Overview",
        "",
        "- SQLite stores local image metadata, paths, descriptions, visual descriptors, and YOLO detection fields.",
        "- OpenCLIP `ViT-B-32` produces normalized 512-dimensional image and text embeddings.",
        "- FAISS `IndexFlatIP` remains the exact local baseline.",
        "- Qdrant stores the same embeddings with payload fields for keywords, style descriptors, and detected objects.",
        "- Unsplash keyword metadata supports broad filtering but can be noisy.",
        "- Visual descriptors capture brightness, contrast, saturation, warmth, and color histograms.",
        "- Style reranking improves visual consistency for style-heavy searches.",
        "- YOLO detected objects add image-derived object evidence.",
        "- Object-aware reranking combines semantic, object, keyword, and style signals without requiring a strict hard filter.",
        "",
        "Application flow:",
        "",
        "```text",
        "User / Streamlit demo",
        "    -> FastAPI",
        "    -> CLIP encoder",
        "    -> Qdrant",
        "    -> metadata/style/object filters",
        "    -> rerankers",
        "    -> image results",
        "```",
        "",
        "## 3. Data and Artifacts",
        "",
        "- Images: 24,916",
        "- CLIP embeddings: 24,916 x 512",
        "- FAISS index vectors: 24,916",
        "- Qdrant collection: `photos`",
        f"- Qdrant points: {stage06_payload.get('qdrant_points', 24916)}",
        f"- YOLO object coverage: {format_float(stage06_payload.get('object_coverage', 0.4583))}",
        f"- Images with detected objects: {stage06_payload.get('images_with_detected_objects', 11418)}",
        f"- Unique detected classes: {stage06_payload.get('unique_detected_objects', 78)}",
        "",
        "## 4. Experiment Timeline",
        "",
        "- `01_clip_baseline`: CLIP-only text retrieval examples.",
        "- `02_style_reranking`: FAISS image retrieval plus style-aware reranking.",
        "- `03_qdrant_backend`: Qdrant backend comparison against FAISS.",
        "- `04_filtered_retrieval`: metadata and style filter evaluation.",
        "- `05_scaled_retrieval_quality`: 24,916-image scale test and visual diagnosis of keyword noise.",
        "- `06_yolo_object_retrieval`: YOLO object payloads, strict object filters, keyword/object combinations, and object-aware reranking.",
        "",
        "## 5. Main Results",
        "",
        "Scale:",
        "",
        "- 24,916 images",
        "- 24,916 embeddings",
        "- 24,916 Qdrant points",
        "",
        "Stage 05:",
        "",
        f"- keyword coverage = {format_float(stage05_payload.get('keyword_coverage', 1.0))}",
        f"- object coverage before YOLO = {format_float(stage05_payload.get('object_coverage', 0.0))}",
        "- visual inspection showed that metadata keyword filters help but can admit noisy object/scene matches.",
        "",
        "Stage 06:",
        "",
        f"- object coverage after YOLO = {format_float(stage06_payload.get('object_coverage', 0.4583))}",
        f"- images with detected objects = {stage06_payload.get('images_with_detected_objects', 11418)}",
        f"- unique detected classes = {stage06_payload.get('unique_detected_objects', 78)}",
        f"- qdrant_object avg relevance = {object_avg_relevance}",
        f"- qdrant_keyword avg relevance = {keyword_avg_relevance}",
        f"- qdrant_object_rerank nDCG@10 = {rerank_ndcg}",
        "",
        "## 6. Qualitative Findings",
        "",
        "- Keyword filters help narrow the corpus, but they are noisy because they depend on external metadata.",
        "- Style reranking improves style consistency, but it does not guarantee object presence.",
        "- YOLO helps object-specific retrieval for COCO-supported classes such as person, dog, cat, bird, and car.",
        "- Object-aware reranking is less brittle than strict filtering because it keeps semantic candidates and promotes object matches instead of excluding all misses.",
        "- Building and architecture remain scene/open-vocabulary limitations because `building` is not a YOLOv8n COCO class.",
        "",
        "Representative visual grids:",
        "",
    ]

    for image_path in selected_visuals:
        lines.extend(
            [
                f"![{image_path.stem}](selected_visualizations/{image_path.name})",
                "",
            ]
        )

    lines.extend(
        [
            "## 7. Application Layer",
            "",
            "The project now includes a local FastAPI API and a Streamlit demo UI. The API exposes health, corpus stats, text search, image search, metadata lookup, and local image-file endpoints. Heavy objects such as the CLIP encoder and Qdrant service are cached through FastAPI dependencies so the model is not reloaded for every request. The Streamlit demo calls the API rather than duplicating retrieval logic.",
            "",
            "## 8. Limitations",
            "",
            "- Automatic relevance labels are weak diagnostics, not human ground truth.",
            "- Manual visual inspection support exists, but stage 06 labels are still incomplete unless `visual_metrics.csv` contains rows.",
            "- YOLOv8n uses the fixed COCO class set and misses open-vocabulary scene concepts.",
            "- Local Qdrant path mode is convenient but not ideal above 20k points.",
            "- CPU YOLO inference is slow for full-corpus extraction.",
            "- The frontend is a local demo, not a production product UI.",
            "",
            "## 9. Future Work",
            "",
            "- Finish manual visual inspection for stage 06.",
            "- Tune the YOLO confidence threshold.",
            "- Store bounding boxes and confidence scores.",
            "- Use Docker/server Qdrant for larger runs.",
            "- Add open-vocabulary detection with GroundingDINO or OWL-ViT.",
            "- Improve and deploy the frontend.",
            "- Deploy the API with a persistent Qdrant server.",
            "",
            "## 10. Conclusion",
            "",
            "The project has evolved from a CLIP-only retrieval baseline into a multi-signal image search system. It combines semantic embeddings, exact FAISS comparison, Qdrant payload filtering, metadata keywords, visual style descriptors, YOLO object detections, and reranking strategies. The resulting system can search by content, style, metadata, and detected object evidence while keeping the experiment trail reproducible.",
            "",
            "## Reproduce Report Assembly",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe experiments\\run_all_experiments.py --assemble-only",
            "```",
        ]
    )

    results_md_path = FINAL_REPORT_DIR / "results.md"
    results_md_path.write_text("\n".join(lines), encoding="utf-8")

    results_pdf_path = FINAL_REPORT_DIR / "results.pdf"
    legacy_pdf_path = PROJECT_ROOT / "experiments" / "results.pdf"
    pandoc_executable = shutil.which("pandoc")
    if pandoc_executable:
        try:
            subprocess.run(
                [pandoc_executable, str(results_md_path), "-o", str(results_pdf_path)],
                check=True,
                cwd=PROJECT_ROOT,
            )
        except subprocess.CalledProcessError:
            if legacy_pdf_path.exists():
                shutil.copy2(legacy_pdf_path, results_pdf_path)
    elif legacy_pdf_path.exists() and not results_pdf_path.exists():
        shutil.copy2(legacy_pdf_path, results_pdf_path)

    print(f"Assembled final report: {results_md_path}")
    print(f"Selected visualizations: {len(selected_visuals)}")


def write_clip_baseline_results() -> None:
    output_path = CLIP_BASELINE_DIR / "results.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "visualization"])
        writer.writeheader()
        for query in BASELINE_QUERIES:
            file_name = f"text_{'_'.join(query.lower().split())}.jpg"
            writer.writerow({"query": query, "visualization": f"visualizations/{file_name}"})


def write_clip_baseline_report() -> None:
    lines = [
        "# CLIP Baseline Report",
        "",
        "This stage captures text-to-image retrieval examples from the FAISS baseline without any Qdrant payload filtering or style reranking.",
        "",
        "## Queries",
        "",
    ]
    for query in BASELINE_QUERIES:
        file_name = f"text_{'_'.join(query.lower().split())}.jpg"
        lines.extend([f"## Query: {query}", "", f"![](visualizations/{file_name})", ""])
    lines.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            "python scripts/visualize_text_search.py --query \"warm cinematic landscape\" --top-k 10",
            "```",
        ]
    )
    (CLIP_BASELINE_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_full_experiment_pipeline() -> None:
    python_executable = resolve_python_executable()

    for query in BASELINE_QUERIES:
        run_command(
            f"baseline visualization: {query}",
            [str(python_executable), "scripts/visualize_text_search.py", "--query", query, "--top-k", "10"],
        )
    write_clip_baseline_results()
    write_clip_baseline_report()

    for image_id in RERANKING_IMAGE_IDS:
        run_command(
            f"reranking comparison: {image_id}",
            [str(python_executable), "scripts/compare_reranking.py", "--image-id", image_id, "--top-k", "5"],
        )

    for description, command in (
        ("style reranking evaluation", ["experiments/scripts/evaluate_style_reranking.py"]),
        ("FAISS vs Qdrant comparison", ["experiments/scripts/compare_faiss_vs_qdrant.py"]),
        ("filtered retrieval comparison", ["experiments/scripts/compare_filtered_retrieval.py"]),
    ):
        run_command(description, [str(python_executable), *command])

    run_command(
        "Qdrant text visualization",
        [str(python_executable), "scripts/visualize_qdrant_text_search.py", "--query", "warm cinematic landscape", "--top-k", "10"],
    )
    run_command(
        "Qdrant image visualization",
        [str(python_executable), "scripts/visualize_qdrant_image_search.py", "--image-id", "oSf8ePoG9NU", "--top-k", "10"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or assemble experiment outputs.")
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Assemble final report from existing artifacts without rerunning expensive experiment steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_experiment_directories()
    if not args.assemble_only:
        run_full_experiment_pipeline()
    assemble_final_report()


if __name__ == "__main__":
    main()
