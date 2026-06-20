import csv
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
    QDRANT_BACKEND_DIR,
    QDRANT_BACKEND_VISUALIZATIONS_DIR,
    STYLE_RERANKING_DIR,
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
FINAL_REPORT_VISUALS = [
    CLIP_BASELINE_VISUALIZATIONS_DIR / "text_warm_cinematic_landscape.jpg",
    CLIP_BASELINE_VISUALIZATIONS_DIR / "text_dark_moody_forest.jpg",
    STYLE_RERANKING_VISUALIZATIONS_DIR / "compare_9U_uCvfpptk.jpg",
    QDRANT_BACKEND_VISUALIZATIONS_DIR / "qdrant_text_warm_cinematic_landscape.jpg",
]


def resolve_python_executable() -> Path:
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return venv_python if venv_python.exists() else Path(sys.executable)


def run_command(description: str, args: list[str]) -> None:
    print(f"[run] {description}", flush=True)
    subprocess.run(args, check=True, cwd=PROJECT_ROOT)


def write_clip_baseline_results() -> None:
    output_path = CLIP_BASELINE_DIR / "results.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query", "visualization"],
        )
        writer.writeheader()
        for query in BASELINE_QUERIES:
            file_name = f"text_{'_'.join(query.lower().split())}.jpg"
            writer.writerow(
                {
                    "query": query,
                    "visualization": f"visualizations/{file_name}",
                }
            )


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
        lines.extend(
            [
                f"## Query: {query}",
                "",
                f"![](visualizations/{file_name})",
                "",
            ]
        )
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


def copy_selected_visualizations() -> None:
    FINAL_REPORT_SELECTED_VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for existing_file in FINAL_REPORT_SELECTED_VISUALIZATIONS_DIR.glob("*"):
        if existing_file.is_file():
            existing_file.unlink()
    for source in FINAL_REPORT_VISUALS:
        if source.exists():
            shutil.copy2(source, FINAL_REPORT_SELECTED_VISUALIZATIONS_DIR / source.name)


def assemble_final_report() -> None:
    copy_selected_visualizations()

    lines = [
        "# Final Results",
        "",
        "This final report links the main artifacts produced by the staged experiment layout.",
        "",
        "## Stage Reports",
        "",
        "- [01 CLIP baseline](../01_clip_baseline/report.md)",
        "- [02 Style reranking](../02_style_reranking/report.md)",
        "- [03 Qdrant backend](../03_qdrant_backend/report.md)",
        "- [04 Filtered retrieval](../04_filtered_retrieval/report.md)",
        "",
        "## Selected Visualizations",
        "",
    ]

    for image_path in sorted(FINAL_REPORT_SELECTED_VISUALIZATIONS_DIR.glob("*.jpg")):
        lines.extend([f"### {image_path.name}", "", f"![](selected_visualizations/{image_path.name})", ""])

    lines.extend(
        [
            "## Reproduce",
            "",
            "```bash",
            "python experiments/run_all_experiments.py",
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
                [
                    pandoc_executable,
                    str(results_md_path),
                    "-o",
                    str(results_pdf_path),
                ],
                check=True,
                cwd=PROJECT_ROOT,
            )
        except subprocess.CalledProcessError:
            if legacy_pdf_path.exists():
                shutil.copy2(legacy_pdf_path, results_pdf_path)
    elif legacy_pdf_path.exists() and not results_pdf_path.exists():
        shutil.copy2(legacy_pdf_path, results_pdf_path)


def main() -> None:
    ensure_experiment_directories()
    python_executable = resolve_python_executable()

    for query in BASELINE_QUERIES:
        run_command(
            f"baseline visualization: {query}",
            [
                str(python_executable),
                "scripts/visualize_text_search.py",
                "--query",
                query,
                "--top-k",
                "10",
            ],
        )
    write_clip_baseline_results()
    write_clip_baseline_report()

    for image_id in RERANKING_IMAGE_IDS:
        run_command(
            f"reranking comparison: {image_id}",
            [
                str(python_executable),
                "scripts/compare_reranking.py",
                "--image-id",
                image_id,
                "--top-k",
                "5",
            ],
        )
    run_command(
        "style reranking evaluation",
        [str(python_executable), "experiments/scripts/evaluate_style_reranking.py"],
    )

    run_command(
        "FAISS vs Qdrant comparison",
        [str(python_executable), "experiments/scripts/compare_faiss_vs_qdrant.py"],
    )

    run_command(
        "filtered retrieval comparison",
        [str(python_executable), "experiments/scripts/compare_filtered_retrieval.py"],
    )

    run_command(
        "Qdrant text visualization",
        [
            str(python_executable),
            "scripts/visualize_qdrant_text_search.py",
            "--query",
            "warm cinematic landscape",
            "--top-k",
            "10",
        ],
    )
    run_command(
        "Qdrant image visualization",
        [
            str(python_executable),
            "scripts/visualize_qdrant_image_search.py",
            "--image-id",
            "oSf8ePoG9NU",
            "--top-k",
            "10",
        ],
    )

    print("[run] final report assembly", flush=True)
    assemble_final_report()


if __name__ == "__main__":
    main()
