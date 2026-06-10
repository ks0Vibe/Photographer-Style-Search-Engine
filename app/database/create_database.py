import sqlite3
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_CSV_PATH = PROJECT_ROOT / "data" / "unsplash-lite" / "metadata.csv"
DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"


def create_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def create_tables(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        image_id TEXT PRIMARY KEY,
        file_path TEXT NOT NULL,

        photo_url TEXT,
        photo_image_url TEXT,
        download_url TEXT,

        width INTEGER,
        height INTEGER,
        aspect_ratio REAL,

        description TEXT,
        ai_description TEXT,

        photographer_username TEXT,
        stats_views INTEGER,
        stats_downloads INTEGER,

        blur_hash TEXT,
        source TEXT,

        brightness REAL,
        contrast REAL,
        saturation REAL,
        warmth REAL,

        color_histogram TEXT,

        clip_embedding_path TEXT,
        embedding_version TEXT,

        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS search_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        query_type TEXT NOT NULL,
        query_text TEXT,

        index_type TEXT NOT NULL,
        search_mode TEXT NOT NULL,

        candidate_k INTEGER,
        top_k INTEGER,

        latency_ms REAL,

        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,

        model_name TEXT,
        embedding_dim INTEGER,

        index_type TEXT,
        index_params TEXT,

        search_mode TEXT,
        candidate_k INTEGER,
        top_k INTEGER,
        reranking_enabled INTEGER,

        precision_at_10 REAL,
        recall_at_10 REAL,
        ndcg_at_10 REAL,

        avg_latency_ms REAL,
        p95_latency_ms REAL,

        index_size_mb REAL,
        ram_usage_mb REAL,

        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()


def import_metadata(conn: sqlite3.Connection) -> None:
    if not METADATA_CSV_PATH.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV_PATH}")

    df = pd.read_csv(METADATA_CSV_PATH)

    required_columns = ["image_id", "file_path"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in metadata.csv: {column}")

    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
        INSERT OR REPLACE INTO images (
            image_id,
            file_path,
            photo_url,
            photo_image_url,
            download_url,
            width,
            height,
            aspect_ratio,
            description,
            ai_description,
            photographer_username,
            stats_views,
            stats_downloads,
            blur_hash,
            source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            row.get("image_id"),
            row.get("file_path"),
            row.get("photo_url"),
            row.get("photo_image_url"),
            row.get("download_url"),
            safe_int(row.get("width")),
            safe_int(row.get("height")),
            safe_float(row.get("aspect_ratio")),
            safe_text(row.get("description")),
            safe_text(row.get("ai_description")),
            safe_text(row.get("photographer_username")),
            safe_int(row.get("stats_views")),
            safe_int(row.get("stats_downloads")),
            safe_text(row.get("blur_hash")),
            safe_text(row.get("source")),
        ))

    conn.commit()


def safe_text(value):
    if pd.isna(value):
        return None
    return str(value)


def safe_int(value):
    if pd.isna(value):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def safe_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def print_database_summary(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM images;")
    image_count = cursor.fetchone()[0]

    cursor.execute("SELECT image_id, file_path, width, height FROM images LIMIT 5;")
    sample_rows = cursor.fetchall()

    print(f"Database created: {DATABASE_PATH}")
    print(f"Images in database: {image_count}")

    print("\nSample rows:")
    for row in sample_rows:
        print(row)


def main() -> None:
    conn = create_connection()

    try:
        create_tables(conn)
        import_metadata(conn)
        print_database_summary(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()