import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Inspecionar ENA
print("=== ENA 2021 (ano de crise) ===")
df_ena = pd.read_excel(os.path.join(BASE_DIR, 'ENA', 'ENA_DIARIO_SUBSISTEMA_2021.xlsx'))
print(df_ena.head(10).to_string())
print(f"\nColunas: {df_ena.columns.tolist()}")
print(f"Shape: {df_ena.shape}")
print(f"Dtypes:\n{df_ena.dtypes}")

print("\n=== ENA 2015 ===")
df_15 = pd.read_excel(os.path.join(BASE_DIR, 'ENA', 'ENA_DIARIO_SUBSISTEMA_2015.xlsx'))
print(df_15.head(5).to_string())
print(f"Colunas: {df_15.columns.tolist()}")

# Inspecionar dados hidrologicos detalhados
print("\n=== DADOS_HIDROLOGICOS_RES_2021 (primeiras linhas) ===")
df_h = pd.read_csv(os.path.join(BASE_DIR, 'dados_hidrologicos', 'DADOS_HIDROLOGICOS_RES_2021.csv'),
                   sep=';', encoding='latin1', nrows=5)
print(df_h.to_string())
print(f"\nColunas: {df_h.columns.tolist()}")

# Inspecionar CARGA_MENSAL
print("\n=== CARGA_MENSAL.parquet ===")
df_carga = pd.read_parquet(os.path.join(BASE_DIR, 'CARGA_MENSAL.parquet'))
print(df_carga.head(5).to_string())
print(f"Colunas: {df_carga.columns.tolist()}")
print(f"Shape: {df_carga.shape}")
