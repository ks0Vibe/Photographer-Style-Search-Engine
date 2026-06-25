import sqlite3
import json
from collections import Counter

DB_PATH = "data/metadata.sqlite"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("Total images:", cur.execute("SELECT COUNT(*) FROM images").fetchone()[0])

# подставь правильное имя колонки, если у тебя она называется не keywords
rows = cur.execute("SELECT keywords FROM images").fetchall()

non_empty = 0
keyword_counts = []
counter = Counter()

for (kw_raw,) in rows:
    if not kw_raw:
        keyword_counts.append(0)
        continue

    try:
        kws = json.loads(kw_raw)
    except Exception:
        kws = [x.strip() for x in str(kw_raw).split(",") if x.strip()]

    if kws:
        non_empty += 1
    keyword_counts.append(len(kws))

    for kw in kws:
        counter[str(kw).lower()] += 1

print("Images with non-empty keywords:", non_empty)
print("Keyword coverage:", non_empty / len(rows))
print("Average keywords per image:", sum(keyword_counts) / len(keyword_counts))
print("Top 30 keywords:")
for kw, count in counter.most_common(30):
    print(kw, count)