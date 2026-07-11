# Vector Optimization Evaluation

## Goal

This experiment evaluates vector optimization options for the CLIP retrieval stack. It compares the original 512-dimensional `float32` vectors against lower-memory representations and measures memory footprint, search latency, and ranking similarity to the baseline.

## Summary

| variant | dim | dtype | vector_memory_25k_mb | estimated_vector_memory_500k_mb | memory_reduction_vs_fp32 | avg_latency_ms | avg_overlap_at_10 | labeled_precision_at_10 | labeled_ndcg_at_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp32_512_baseline | 512 | float32 | 48.6641 | 976.5625 | 1.0000 | 1.5651 | 1.0000 | 0.9500 | 0.9307 |
| float16_512 | 512 | float16 | 24.3320 | 488.2812 | 2.0000 | 31.2003 | 0.9964 | 0.9500 | 0.9263 |
| int8_per_vector_512 | 512 | int8+scale | 12.2611 | 246.0480 | 3.9690 | 32.1704 | 0.8750 | 0.9397 | 0.9405 |
| pca256_fp32 | 256 | float32 | 24.3320 | 488.2812 | 2.0000 | 0.9615 | 0.0571 | 0.7500 | 0.7500 |
| pca256_fp16 | 256 | float16 | 12.1660 | 244.1406 | 4.0000 | 15.3456 | 0.0571 | 0.7500 | 0.7500 |
| pca128_fp32 | 128 | float32 | 12.1660 | 244.1406 | 4.0000 | 0.6358 | 0.0571 | 1.0000 | 1.0000 |
| pca128_fp16 | 128 | float16 | 6.0830 | 122.0703 | 8.0000 | 8.0047 | 0.0571 | 1.0000 | 1.0000 |

## Docker Measurement

- Docker stats were not available during this run.
- Error: `Command '['docker', 'stats', '--no-stream', '--format', '{{json .}}', 'photographer-style-qdrant']' returned non-zero exit status 1.`

To capture container-level memory during defense preparation, run:

```powershell
docker compose up -d qdrant
docker stats --no-stream photographer-style-qdrant
```

## Interpretation

- `float16_512` keeps the same embedding dimension and should preserve ranking almost exactly while halving vector storage.
- `int8_per_vector_512` gives about 4x vector-memory reduction, but pure Python dequantization can increase latency; in production this should be delegated to Qdrant scalar quantization.
- PCA variants reduce dimensionality. They lower memory and can improve cache behavior, but ranking overlap and human-label metrics show the quality trade-off.
- The 500k memory estimate scales only vector payload memory. Full Docker/Qdrant memory also includes HNSW graph, payload indexes, metadata, WAL, allocator overhead, and container runtime overhead.

## Outputs

- `vector_optimization_summary.csv`: aggregated variant comparison.
- `vector_optimization_query_results.csv`: per-query latency, overlap, and labeled metrics.
- `artifact_sizes.json`: file-level sizes for generated optimized vector artifacts.
- `docker_stats.json`: container memory snapshot when Docker is available.

Per-query rows generated: 196
