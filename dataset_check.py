import pandas as pd

df = pd.read_csv(
    "data/unsplash-lite/photos.csv000",
    sep="\t",
    nrows=5
)

print(df.columns.tolist())

print("\nFirst row:")
print(df.iloc[0])