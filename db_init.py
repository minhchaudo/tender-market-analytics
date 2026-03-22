import pandas as pd
import sqlite3

csv_path = "kit_test_df_clean.csv"
sqlite_path = "products.db"
table_name = "products"
fts_table = "products_fts"

df = pd.read_csv(csv_path)
COLUMNS = df.columns

NUMERIC_FIELDS = {"quantity", "unit_price", "total_price"}
DATE_FIELDS = {"posting_date", "closing_date"}
TEXT_FIELDS = {col for col in COLUMNS if col not in NUMERIC_FIELDS | DATE_FIELDS}

# df = df.rename(columns=COLUMN_MAP)

for col in NUMERIC_FIELDS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in DATE_FIELDS:
    dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    df[col] = dt.dt.normalize().dt.strftime("%Y-%m-%d %H:%M:%S")
    df[col] = df[col].where(dt.notna(), None)

df.insert(0, "id", range(1, len(df) + 1))

conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()

cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
cur.execute(f'DROP TABLE IF EXISTS "{fts_table}"')
cur.execute(f'DROP TRIGGER IF EXISTS "{table_name}_ai"')
cur.execute(f'DROP TRIGGER IF EXISTS "{table_name}_ad"')
cur.execute(f'DROP TRIGGER IF EXISTS "{table_name}_au"')

df.to_sql(table_name, conn, if_exists="replace", index=False)

for col in NUMERIC_FIELDS | DATE_FIELDS:
    cur.execute(
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{col}" '
        f'ON "{table_name}"("{col}")'
    )

fts_cols_sql = ",\n    ".join(f'"{col}"' for col in TEXT_FIELDS)

cur.execute(f"""
CREATE VIRTUAL TABLE "{fts_table}" USING fts5(
    {fts_cols_sql},
    content="{table_name}",
    content_rowid="id",
    tokenize='unicode61 remove_diacritics 2'
)
""")

cur.execute(f'INSERT INTO "{fts_table}"("{fts_table}") VALUES ("rebuild")')

old_cols = ", ".join(f'old."{col}"' for col in TEXT_FIELDS)
new_cols = ", ".join(f'new."{col}"' for col in TEXT_FIELDS)
fts_alias_cols = ", ".join(f'"{col}"' for col in TEXT_FIELDS)

cur.executescript(f"""
CREATE TRIGGER "{table_name}_ai" AFTER INSERT ON "{table_name}" BEGIN
    INSERT INTO "{fts_table}"(rowid, {fts_alias_cols})
    VALUES (new."id", {new_cols});
END;

CREATE TRIGGER "{table_name}_ad" AFTER DELETE ON "{table_name}" BEGIN
    INSERT INTO "{fts_table}"("{fts_table}", rowid, {fts_alias_cols})
    VALUES ('delete', old."id", {old_cols});
END;

CREATE TRIGGER "{table_name}_au" AFTER UPDATE ON "{table_name}" BEGIN
    INSERT INTO "{fts_table}"("{fts_table}", rowid, {fts_alias_cols})
    VALUES ('delete', old."id", {old_cols});
    INSERT INTO "{fts_table}"(rowid, {fts_alias_cols})
    VALUES (new."id", {new_cols});
END;
""")

conn.commit()
conn.close()

print(
    f"Loaded {len(df)} rows into {sqlite_path} -> "
    f"table '{table_name}' with FTS table '{fts_table}'"
)
