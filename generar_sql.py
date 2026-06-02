import sqlite3
import pandas as pd

conn = sqlite3.connect('historial.db')
df = pd.read_csv('DATASET.csv')
df.to_sql('dataset', conn, if_exists='replace', index=False)

with open('dataset.sql', 'w', encoding='utf-8') as f:
    f.write("-- ============================================\n")
    f.write("-- SISTEMA DE PREDIAGNOSTICO - DATASET\n")
    f.write("-- Generado desde DATASET.csv\n")
    f.write("-- ============================================\n\n")

    f.write("CREATE TABLE IF NOT EXISTS dataset (\n")
    for col in df.columns[:-1]:
        f.write(f"    {col} TEXT,\n")
    f.write(f"    {df.columns[-1]} TEXT\n")
    f.write(");\n\n")

    for _, row in df.iterrows():
        values = []
        for v in row:
            if pd.isna(v):
                values.append('NULL')
            else:
                vstr = str(v).replace("'", "''")
                values.append(f"'{vstr}'")
        f.write(f"INSERT INTO dataset VALUES ({', '.join(values)});\n")

conn.close()

print(f"dataset.sql generado con {len(df)} registros")
