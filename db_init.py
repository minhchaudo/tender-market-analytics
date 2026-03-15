import pandas as pd
import sqlite3

# Input/output paths
excel_path = "Market_survey.xlsx"
sqlite_path = "products.db"
table_name = "products"

# Read Excel
df = pd.read_excel(excel_path)
df["Số lượng"] = pd.to_numeric(df["Số lượng"], errors="coerce")
df["Đơn giá trúng thầu"] = pd.to_numeric(df["Đơn giá trúng thầu"], errors="coerce")
df["Thành tiền"] = pd.to_numeric(df["Thành tiền"], errors="coerce")

# Write to SQLite
conn = sqlite3.connect(sqlite_path)
df.to_sql(table_name, conn, if_exists="replace", index=False)
conn.close()

print(f"Loaded {len(df)} rows into {sqlite_path} -> table '{table_name}'")