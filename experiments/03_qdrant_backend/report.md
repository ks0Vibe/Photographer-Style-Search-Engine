# FAISS vs Qdrant Comparison

This stage compares the baseline FAISS backend against the Qdrant backend under unfiltered retrieval so the advanced backend can be judged against a stable exact-search reference.

## Setup

- Query set: 30 image queries and 5 text queries
- Top-K: 10
- FAISS mode: Flat cosine search
- Qdrant mode: local persistent cosine collection with payloads

## Results

- FAISS average latency: 47.48 ms
- FAISS p95 latency: 54.53 ms
- Qdrant average latency: 50.04 ms
- Qdrant p95 latency: 56.63 ms
- Average overlap@10: 1.000
- Top-1 consistency: 1.000
- FAISS index size: 2048045 bytes
- Qdrant local data size: 6222343 bytes

## Interpretation

- FAISS stays as the exact local baseline.
- Qdrant remains close enough to baseline behavior for unfiltered search while adding database features that FAISS does not provide.
- The visualizations in `visualizations/` show sample Qdrant retrieval outputs produced by the same backend.

## Reproduce

```bash
python experiments/scripts/compare_faiss_vs_qdrant.py
```