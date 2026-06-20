# Experiments Index

| Experiment | Goal | Command to Reproduce | Output Files | Status |
| --- | --- | --- | --- | --- |
| `01_clip_baseline` | Generate baseline FAISS text-to-image visualizations from CLIP-only retrieval. | `.\.venv\Scripts\python.exe scripts/visualize_text_search.py --query "warm cinematic landscape" --top-k 10` | `01_clip_baseline/report.md`, `01_clip_baseline/results.csv`, `01_clip_baseline/visualizations/*.jpg` | Active |
| `02_style_reranking` | Measure whether style reranking improves visual similarity over the FAISS baseline. | `.\.venv\Scripts\python.exe experiments/scripts/evaluate_style_reranking.py` | `02_style_reranking/report.md`, `02_style_reranking/metrics.csv`, `02_style_reranking/evaluation_output.txt`, `02_style_reranking/visualizations/*.jpg` | Active |
| `03_qdrant_backend` | Compare the Qdrant backend against the FAISS baseline and store sample Qdrant visualizations. | `.\.venv\Scripts\python.exe experiments/scripts/compare_faiss_vs_qdrant.py` | `03_qdrant_backend/report.md`, `03_qdrant_backend/faiss_vs_qdrant_results.csv`, `03_qdrant_backend/faiss_vs_qdrant_results.md`, `03_qdrant_backend/visualizations/*.jpg` | Active |
| `04_filtered_retrieval` | Evaluate Qdrant payload filtering and reranking behavior on text retrieval. | `.\.venv\Scripts\python.exe experiments/scripts/compare_filtered_retrieval.py` | `04_filtered_retrieval/report.md`, `04_filtered_retrieval/filtered_retrieval_results.csv`, `04_filtered_retrieval/filtered_retrieval_results.md`, `04_filtered_retrieval/visualizations/` | Active |
| `final_report` | Assemble a single roll-up report that links the staged outputs and selected images. | `.\.venv\Scripts\python.exe experiments/run_all_experiments.py` | `final_report/results.md`, `final_report/results.pdf`, `final_report/selected_visualizations/` | Active |

## Full Run

```bash
.\.venv\Scripts\python.exe experiments/run_all_experiments.py
```
