import sqlite3
from pathlib import Path

DB_PATH = Path("nb/src/data/knowledge_graph.db")

if not DB_PATH.exists():
    print("❌ DATABASE NOT FOUND.")
else:
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        print("--- Tables in DB ---")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(tables)

        if ('nutrition_facts',) in tables:
            print("\n--- Columns in nutrition_facts table ---")
            cursor.execute("PRAGMA table_info(nutrition_facts);")
            columns = cursor.fetchall()
            for col in columns:
                print(col)
    finally:
        conn.close() 