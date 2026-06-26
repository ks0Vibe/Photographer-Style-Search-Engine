# Synthetic 500k Vector Scale Evaluation

## 1. Goal

This experiment addresses the 500k-object scale requirement by adding synthetic vector objects derived from real CLIP embeddings.

Real visual corpus: 24,916 downloaded images. Synthetic scale corpus: 500,000 generated vector objects. Purpose of synthetic corpus: scalability and indexing benchmark, not visual relevance evaluation.

## 2. Why Synthetic Objects

Generating and downloading 500,000 real images is not feasible under the local hardware, storage, and time constraints of this project. Synthetic vectors are therefore used only to test vector database indexing, storage, filtering, and search latency at the required object scale.

## 3. Generation Method

- Real normalized CLIP embeddings are used as anchors.
- Each synthetic vector is generated with Gaussian perturbation.
- A configurable subset also mixes in a neighboring real embedding.
- Every generated vector is L2-normalized and stored as `float32`.
- Payload templates are copied from source images, including keywords and detected objects when available.
- Every payload includes `is_synthetic = true` and `synthetic_generation = clip_embedding_perturbation_v1`.
- `file_path` is only a proxy reference to the source real image for debugging.

## 4. Dataset Statistics

| stat | value |
| --- | --- |
| real visual corpus | 24,916 downloaded images |
| synthetic vector corpus | 500,000 generated vector objects |
| total benchmark scale | 500,000 synthetic objects |
| embedding dim | 512 |
| vector dtype | float32 |
| approx vector memory MB | 976.56 |

## 5. Qdrant Collection

| stat | value |
| --- | --- |
| collection name | photos_synthetic_500k |
| Qdrant mode | server |
| distance metric | Cosine |
| vector size | 512 |
| upload time seconds | 317.2120 |
| storage size MB | 1776.1481 |

Docker/server Qdrant is preferred for the 500k synthetic benchmark. Local path mode is supported for reproducibility, but it is less suitable for large collections.

## 6. Search Benchmark

| mode | query_count | collection_size | candidate_pool_size | qdrant_mode | search_latency_ms_avg | search_latency_ms_p50 | search_latency_ms_p95 | search_latency_ms_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qdrant_synthetic_keyword_filter | 8 | 500000 | 100 | server | 202.5247 | 202.5007 | 220.5080 | 572.3812 |
| qdrant_synthetic_object_filter | 8 | 500000 | 100 | server | 355.8509 | 351.4675 | 385.1083 | 588.9145 |
| qdrant_synthetic_semantic | 8 | 500000 | 100 | server | 31.3680 | 31.3564 | 44.0736 | 46.8979 |

Detailed per-query latency rows are saved in `synthetic_benchmark_results.csv`. Aggregated mode summaries are saved in `synthetic_latency_summary.csv`.

## 7. Interpretation

The synthetic objects validate that the vector database can hold and search the required 500k-object scale. Keyword and object filters are checked for operational behavior against copied payload fields. Real-image retrieval quality evaluation remains based on the 24,916 downloaded images, and synthetic results should not be used for visual relevance conclusions.

## 8. Limitations

- Synthetic vectors are not independent real photographs.
- Metadata payloads are copied from source images.
- Visual relevance cannot be judged from synthetic duplicates or perturbations.
- A real 500k image dataset would be stronger but requires much more storage, download time, and compute.

## 9. Conclusion

The project now includes a real visual retrieval corpus for quality experiments and a synthetic 500k vector corpus for scalability, indexing, latency, and hardware requirement evaluation.
