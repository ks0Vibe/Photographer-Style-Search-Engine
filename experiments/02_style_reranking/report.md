# Style Reranking Report

This stage measures whether style-aware reranking produces neighbors that are visually closer to the query than the FAISS semantic baseline alone.

- Query sample size: 30
- Candidate pool size before reranking: 100
- Evaluated top-k per query: 10

## Metrics

| Metric | CLIP only | Reranked | Improvement |
| --- | ---: | ---: | ---: |
| brightness | 0.1698 | 0.0924 | 0.0774 |
| contrast | 0.0745 | 0.0577 | 0.0167 |
| saturation | 0.1821 | 0.1239 | 0.0582 |
| warmth | 0.0510 | 0.0399 | 0.0111 |

Improvement is baseline difference minus reranked difference, so positive values mean the reranker moved the results closer to the query style.

## Visual Comparisons

## Query Image: 9U_uCvfpptk

![](visualizations/compare_9U_uCvfpptk.jpg)

## Query Image: 9wTWFyInJ4Y

![](visualizations/compare_9wTWFyInJ4Y.jpg)

## Query Image: 39DcBUbYZP4

![](visualizations/compare_39DcBUbYZP4.jpg)

## Query Image: A-G8q9zorGs

![](visualizations/compare_A-G8q9zorGs.jpg)


## Reproduce

```bash
python experiments/scripts/evaluate_style_reranking.py
```