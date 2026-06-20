# CLIP Baseline Report

This stage captures text-to-image retrieval examples from the FAISS baseline without any Qdrant payload filtering or style reranking.

## Queries

## Query: warm cinematic landscape

![](visualizations/text_warm_cinematic_landscape.jpg)

## Query: dark moody forest

![](visualizations/text_dark_moody_forest.jpg)

## Query: cold snowy mountain

![](visualizations/text_cold_snowy_mountain.jpg)

## Query: bright tropical beach

![](visualizations/text_bright_tropical_beach.jpg)

## Query: minimal street photography

![](visualizations/text_minimal_street_photography.jpg)

## Reproduce

```bash
python scripts/visualize_text_search.py --query "warm cinematic landscape" --top-k 10
```