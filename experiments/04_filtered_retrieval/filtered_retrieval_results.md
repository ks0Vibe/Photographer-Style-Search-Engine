# Filtered Retrieval Comparison

This stage exercises the Qdrant payload features that are not available in the FAISS baseline: keyword filters, style-range filters, and optional reranking on the filtered candidate set.

Compared modes:

- Qdrant semantic search only
- Qdrant semantic search + keyword filter
- Qdrant semantic search + style filter
- Qdrant semantic search + reranking

| Query | Mode | Filter | Count | Top-1 | Top-1 Score | Avg Score | Avg Brightness | Avg Warmth |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| warm cinematic landscape | semantic_only |  | 10 | YAt2KX0JVdE | 0.2674 | 0.2619 | 0.3580 | 0.5002 |
| warm cinematic landscape | keyword_filter | nature | 10 | YAt2KX0JVdE | 0.2674 | 0.2619 | 0.3580 | 0.5002 |
| warm cinematic landscape | style_filter | min_warmth=0.55 | 10 | frCSu-B5Mxg | 0.2549 | 0.2467 | 0.4553 | 0.5953 |
| warm cinematic landscape | rerank | style_rerank | 10 | fHK6kIQmtTw | 0.3748 | 0.3653 | 0.4748 | 0.5412 |
| dark moody forest | semantic_only |  | 10 | odfhHtD3igY | 0.3052 | 0.2947 | 0.2730 | 0.4923 |
| dark moody forest | keyword_filter | nature | 10 | odfhHtD3igY | 0.3052 | 0.2947 | 0.2730 | 0.4923 |
| dark moody forest | style_filter | max_brightness=0.4 | 10 | odfhHtD3igY | 0.3052 | 0.2935 | 0.2660 | 0.4889 |
| dark moody forest | rerank | style_rerank | 10 | hjGJFIXW8QM | 0.4061 | 0.3966 | 0.2814 | 0.4959 |
| street photography | semantic_only |  | 10 | fi9PSKk7UyE | 0.2587 | 0.2484 | 0.3668 | 0.5252 |
| street photography | keyword_filter | person | 10 | 8sbwihARadA | 0.2488 | 0.2396 | 0.4455 | 0.5135 |
| street photography | style_filter | max_brightness=0.5 | 10 | fi9PSKk7UyE | 0.2587 | 0.2469 | 0.3382 | 0.5279 |
| street photography | rerank | style_rerank | 10 | fi9PSKk7UyE | 0.2587 | 0.2484 | 0.3668 | 0.5252 |
| portrait | semantic_only |  | 10 | Jr_DkO3Ow14 | 0.2657 | 0.2505 | 0.3890 | 0.5151 |
| portrait | keyword_filter | person | 10 | dwXKJSwJe7Q | 0.2473 | 0.2290 | 0.4807 | 0.5337 |
| portrait | style_filter | min_warmth=0.45 | 10 | Jr_DkO3Ow14 | 0.2657 | 0.2505 | 0.3890 | 0.5151 |
| portrait | rerank | style_rerank | 10 | Jr_DkO3Ow14 | 0.2657 | 0.2505 | 0.3890 | 0.5151 |
| beach sunset | semantic_only |  | 10 | JTEi1fWSygE | 0.2855 | 0.2743 | 0.3682 | 0.5571 |
| beach sunset | keyword_filter | water | 10 | JTEi1fWSygE | 0.2855 | 0.2725 | 0.4175 | 0.5567 |
| beach sunset | style_filter | min_warmth=0.6,min_saturation=0.45 | 10 | PzytfAmbOQM | 0.2727 | 0.2511 | 0.3529 | 0.6653 |
| beach sunset | rerank | style_rerank | 10 | ZGpi37nNzOY | 0.4312 | 0.4109 | 0.4220 | 0.6666 |

## Reproduce

```bash
python experiments/scripts/compare_filtered_retrieval.py
```