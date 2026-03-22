import pandas as pd
import sqlite3

excel_path = "Market_survey.xlsx"
sqlite_path = "products.db"
table_name = "products"
fts_table = "products_fts"

COLUMN_MAP = {
    "Tên gói thầu": "bid_name",
    "Chủ đầu tư": "investor",
    "Địa điểm": "location",
    "Nhà thầu trúng thầu": "winner",
    "Tên hàng hoá": "item_name",
    "Mặt hàng": "category",
    "Nhà sản xuất": "manufacturer",
    "Xuất xứ": "origin",
    "Đơn vị": "unit",
    "Số lượng": "quantity",
    "Đơn giá trúng thầu": "unit_price",
    "Thành tiền": "total_price",
    "Thời điểm đăng tải": "posting_time",
    "Thời điểm đóng thầu": "closing_time",
}

NUMERIC_SOURCE_FIELDS = {"Số lượng", "Đơn giá trúng thầu", "Thành tiền"}
DATE_SOURCE_FIELDS = {"Thời điểm đăng tải", "Thời điểm đóng thầu"}
TEXT_SOURCE_FIELDS = [col for col in COLUMN_MAP if col not in NUMERIC_SOURCE_FIELDS | DATE_SOURCE_FIELDS]

df = pd.read_excel(excel_path)

df = df.rename(columns=COLUMN_MAP)

for src_col in NUMERIC_SOURCE_FIELDS:
    col = COLUMN_MAP[src_col]
    df[col] = pd.to_numeric(df[col], errors="coerce")

for src_col in DATE_SOURCE_FIELDS:
    col = COLUMN_MAP[src_col]
    dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    df[col] = dt.dt.normalize().dt.strftime("%Y-%m-%d %H:%M:%S")
    df[col] = df[col].where(dt.notna(), None)

df.insert(0, "id", range(1, len(df) + 1))

NUMERIC_FIELDS = [COLUMN_MAP[col] for col in NUMERIC_SOURCE_FIELDS]
DATE_FIELDS = [COLUMN_MAP[col] for col in DATE_SOURCE_FIELDS]
TEXT_FIELDS = [COLUMN_MAP[col] for col in TEXT_SOURCE_FIELDS]

conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()

cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
cur.execute(f'DROP TABLE IF EXISTS "{fts_table}"')
cur.execute(f'DROP TRIGGER IF EXISTS "{table_name}_ai"')
cur.execute(f'DROP TRIGGER IF EXISTS "{table_name}_ad"')
cur.execute(f'DROP TRIGGER IF EXISTS "{table_name}_au"')

df.to_sql(table_name, conn, if_exists="replace", index=False)

for col in NUMERIC_FIELDS + DATE_FIELDS:
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