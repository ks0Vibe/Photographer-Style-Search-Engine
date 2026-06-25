import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "metadata.sqlite"

DETECTION_COLUMNS = {
    "detected_objects": "TEXT DEFAULT '[]'",
    "detection_model": "TEXT",
    "detection_updated_at": "TEXT",
}


def get_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(images);").fetchall()
    return {str(row[1]) for row in rows}


def main() -> None:
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")

    conn = sqlite3.connect(DATABASE_PATH)
    try:
        existing_columns = get_columns(conn)
        for column, definition in DETECTION_COLUMNS.items():
            if column in existing_columns:
                print(f"Column already exists: images.{column}")
                continue
            conn.execute(f"ALTER TABLE images ADD COLUMN {column} {definition};")
            print(f"Added column: images.{column}")
        conn.commit()

        final_columns = get_columns(conn)
        print()
        print("Detection-related schema fields:")
        for column in DETECTION_COLUMNS:
            print(f"- {column}: {'present' if column in final_columns else 'missing'}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
