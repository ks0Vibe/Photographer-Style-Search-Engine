from pathlib import Path
from collections import Counter
import pandas as pd

KEYWORDS_PATH = Path("data/unsplash-lite/keywords.csv000")
METADATA_PATH = Path("data/unsplash-lite/metadata.csv")

if not KEYWORDS_PATH.exists():
    raise FileNotFoundError(f"Not found: {KEYWORDS_PATH}")

def read_csv_smart(path: Path) -> pd.DataFrame:
    # Unsplash Lite files are often tab-separated.
    # This fallback makes the script robust if separator differs.
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=None, engine="python")
    return df

keywords_df = read_csv_smart(KEYWORDS_PATH)

print("Keywords CSV shape:", keywords_df.shape)
print("Keywords CSV columns:")
print(list(keywords_df.columns))
print()

# Try to infer columns
photo_col_candidates = ["photo_id", "id", "image_id"]
keyword_col_candidates = ["keyword", "name", "term", "value"]

photo_col = None
keyword_col = None

for col in photo_col_candidates:
    if col in keywords_df.columns:
        photo_col = col
        break

for col in keyword_col_candidates:
    if col in keywords_df.columns:
        keyword_col = col
        break

if photo_col is None or keyword_col is None:
    print("Could not infer photo/keyword columns automatically.")
    print("Please send me the printed column list above.")
    raise SystemExit

keywords_df[keyword_col] = keywords_df[keyword_col].astype(str).str.lower().str.strip()

total_keyword_rows = len(keywords_df)
unique_photos_with_keywords = keywords_df[photo_col].nunique()
unique_keywords = keywords_df[keyword_col].nunique()

print("Total keyword rows:", total_keyword_rows)
print("Photos with at least one keyword:", unique_photos_with_keywords)
print("Unique keywords:", unique_keywords)

if METADATA_PATH.exists():
    metadata_df = read_csv_smart(METADATA_PATH)
    print("Metadata shape:", metadata_df.shape)
    print("Metadata columns:")
    print(list(metadata_df.columns))

    metadata_id_col = None
    for col in ["photo_id", "id", "image_id"]:
        if col in metadata_df.columns:
            metadata_id_col = col
            break

    if metadata_id_col is not None:
        total_images = len(metadata_df)
        coverage = unique_photos_with_keywords / total_images
        print("Keyword coverage over metadata images:", coverage)

print()
print("Top 50 keywords:")
counter = Counter(keywords_df[keyword_col].dropna().tolist())

for keyword, count in counter.most_common(50):
    print(f"{keyword}: {count}")