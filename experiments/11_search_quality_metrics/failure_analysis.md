# Failure Analysis

## Lowest-Performing Queries

- `Q006` (semantic_text): food photography with warm light - average nDCG@10=0.738.
- `Q012` (style_text): cinematic teal and orange - average nDCG@10=0.740.
- `Q016` (style_text): saturated colorful street scene - average nDCG@10=0.777.
- `Q026` (mixed_text): dark forest with animal - average nDCG@10=0.822.
- `Q005` (semantic_text): cold blue mountain scene - average nDCG@10=0.822.
- `Q024` (object_text): train in station - average nDCG@10=0.841.
- `Q023` (object_text): cup on table - average nDCG@10=0.843.
- `Q027` (mixed_text): bright beach scene with dog - average nDCG@10=0.879.
- `Q001` (semantic_text): warm cinematic landscape - average nDCG@10=0.904.
- `Q007` (semantic_text): quiet city street at dawn - average nDCG@10=0.913.

## Largest System Differences

- `Q012` (style_text): gap=0.270; best `faiss_baseline,qdrant_filtered,qdrant_object_rerank,qdrant_semantic`=0.794, worst `faiss_style_rerank`=0.524. Query: cinematic teal and orange
- `Q027` (mixed_text): gap=0.261; best `qdrant_object_rerank`=1.000, worst `faiss_style_rerank`=0.739. Query: bright beach scene with dog
- `Q023` (object_text): gap=0.250; best `qdrant_filtered,qdrant_object_rerank`=0.993, worst `faiss_baseline,faiss_style_rerank,qdrant_semantic`=0.743. Query: cup on table
- `Q026` (mixed_text): gap=0.222; best `faiss_style_rerank`=1.000, worst `faiss_baseline,qdrant_filtered,qdrant_object_rerank,qdrant_semantic`=0.778. Query: dark forest with animal
- `Q006` (semantic_text): gap=0.213; best `faiss_style_rerank`=0.908, worst `faiss_baseline,qdrant_filtered,qdrant_object_rerank,qdrant_semantic`=0.695. Query: food photography with warm light
- `Q024` (object_text): gap=0.197; best `qdrant_filtered`=0.980, worst `faiss_baseline,faiss_style_rerank,qdrant_semantic`=0.783. Query: train in station
- `Q005` (semantic_text): gap=0.193; best `faiss_style_rerank`=0.977, worst `faiss_baseline,qdrant_filtered,qdrant_object_rerank,qdrant_semantic`=0.784. Query: cold blue mountain scene
- `Q016` (style_text): gap=0.134; best `qdrant_filtered`=0.880, worst `faiss_baseline,qdrant_object_rerank,qdrant_semantic`=0.746. Query: saturated colorful street scene
- `Q022` (object_text): gap=0.128; best `qdrant_filtered,qdrant_object_rerank`=1.000, worst `faiss_baseline,faiss_style_rerank,qdrant_semantic`=0.872. Query: boat on water
- `Q020` (object_text): gap=0.082; best `faiss_baseline,qdrant_semantic`=0.992, worst `faiss_style_rerank`=0.910. Query: bicycle in urban street

## Reranking Regressions

- `Q012` (style_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.524 vs 0.794, delta=-0.270). Query: cinematic teal and orange
- `Q027` (mixed_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.739 vs 0.833, delta=-0.093). Query: bright beach scene with dog
- `Q020` (object_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.910 vs 0.992, delta=-0.082). Query: bicycle in urban street
- `Q004` (semantic_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.925 vs 0.998, delta=-0.073). Query: minimal architecture
- `Q001` (semantic_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.865 vs 0.914, delta=-0.049). Query: warm cinematic landscape
- `Q015` (style_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.947 vs 0.993, delta=-0.046). Query: bright airy minimal style
- `Q003` (semantic_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.961 vs 1.000, delta=-0.039). Query: vibrant summer portrait
- `Q010` (style_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.984 vs 1.000, delta=-0.016). Query: low contrast pastel tones
- `Q002` (semantic_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.986 vs 1.000, delta=-0.014). Query: dark moody forest
- `Q014` (style_text): `faiss_style_rerank` lowered nDCG@10 versus `faiss_baseline` (0.984 vs 0.993, delta=-0.009). Query: dark low key portrait lighting
- `Q010` (style_text): `qdrant_filtered` lowered nDCG@10 versus `qdrant_semantic` (0.991 vs 1.000, delta=-0.009). Query: low contrast pastel tones
- `Q020` (object_text): `qdrant_filtered` lowered nDCG@10 versus `qdrant_semantic` (0.991 vs 0.992, delta=-0.001). Query: bicycle in urban street
- `Q020` (object_text): `qdrant_object_rerank` lowered nDCG@10 versus `qdrant_semantic` (0.991 vs 0.992, delta=-0.001). Query: bicycle in urban street

## Query-Type Weaknesses

- `faiss_baseline` underperforms its overall nDCG@10 on `mixed_text` (0.884 vs overall 0.913).
- `qdrant_filtered` underperforms its overall nDCG@10 on `mixed_text` (0.936 vs overall 0.948).
- `qdrant_object_rerank` underperforms its overall nDCG@10 on `mixed_text` (0.931 vs overall 0.939).
- `qdrant_semantic` underperforms its overall nDCG@10 on `mixed_text` (0.884 vs overall 0.913).
- `faiss_baseline` underperforms its overall nDCG@10 on `object_text` (0.910 vs overall 0.913).
- `faiss_style_rerank` underperforms its overall nDCG@10 on `object_text` (0.899 vs overall 0.916).
- `qdrant_semantic` underperforms its overall nDCG@10 on `object_text` (0.910 vs overall 0.913).
- `faiss_baseline` underperforms its overall nDCG@10 on `semantic_text` (0.912 vs overall 0.913).
- `qdrant_filtered` underperforms its overall nDCG@10 on `semantic_text` (0.912 vs overall 0.948).
- `qdrant_object_rerank` underperforms its overall nDCG@10 on `semantic_text` (0.912 vs overall 0.939).
- `qdrant_semantic` underperforms its overall nDCG@10 on `semantic_text` (0.912 vs overall 0.913).
- `faiss_style_rerank` underperforms its overall nDCG@10 on `style_text` (0.898 vs overall 0.916).
- `qdrant_object_rerank` underperforms its overall nDCG@10 on `style_text` (0.933 vs overall 0.939).

## Manual Label Comments Used

- `Q001` `faiss_baseline` rank 7 image `Hr8gz4OE-Cs`: not warm
- `Q001` `faiss_baseline` rank 10 image `S95EQl9CzNE`: warm, but not so warm to be 2
- `Q003` `faiss_baseline` rank 7 image `_72NMJyq6mU`: not so vibrant
- `Q005` `faiss_baseline` rank 1 image `8gBUx7Xmsqo`: no moutain
- `Q005` `faiss_baseline` rank 7 image `TEND07a9xDg`: no explicit mountain
- `Q005` `faiss_baseline` rank 9 image `aJE6CJZfPxk`: not blue
- `Q005` `faiss_baseline` rank 10 image `cUr8oSVN3NE`: no mountain
- `Q006` `faiss_baseline` rank 1 image `IeRX8XZZBzc`: not warm
- `Q006` `faiss_baseline` rank 2 image `DWGBY2Wuqeo`: not warm
- `Q006` `faiss_baseline` rank 3 image `OSwea3yxjT0`: not warm

## Warnings

- None
