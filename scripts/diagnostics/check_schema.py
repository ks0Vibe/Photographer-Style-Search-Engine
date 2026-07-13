import sqlite3


DB_PATH = "data/metadata.sqlite"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("Tables:")
tables = cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

for (table_name,) in tables:
    print(f"\n=== {table_name} ===")
    columns = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    for col in columns:
        print(col[1], col[2])

conn.close()
