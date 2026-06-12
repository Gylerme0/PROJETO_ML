import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def build_features():
    print("Iniciando engenharia de variáveis...")
    in_path = os.path.join(PROC_DIR, 'dados_basicos.parquet')
    df = pd.read_parquet(in_path)

    df['Data'] = pd.to_datetime(df['Data'])
    
    cols_num = [c for c in df.columns if c not in ['Data','DataRef','Target_Financeiro']]
    df[cols_num] = df[cols_num].interpolate(method='linear', limit_direction='both')

    df['DiaDoAno_sin'] = np.sin(2 * np.pi * df['Data'].dt.dayofyear / 365)
    df['DiaDoAno_cos'] = np.cos(2 * np.pi * df['Data'].dt.dayofyear / 365)
    df['MesHidrologico'] = ((df['Data'].dt.month - 10) % 12) + 1  
    df['MesHid_sin'] = np.sin(2 * np.pi * df['MesHidrologico'] / 12)
    df['MesHid_cos'] = np.cos(2 * np.pi * df['MesHidrologico'] / 12)

    for col in [c for c in df.columns if c.startswith('Vol_')]:
        df[f'{col}_roll30d'] = df[col].rolling(30, min_periods=15).mean()
        df[f'{col}_roll90d'] = df[col].rolling(90, min_periods=45).mean()
        df[f'{col}_roll365d'] = df[col].rolling(365, min_periods=180).mean()
        df[f'{col}_anomalia_1ano'] = (df[col] - df[f'{col}_roll365d']) / (df[f'{col}_roll365d'] + 1e-6)

    for col in [c for c in df.columns if 'pctMLT' in c]:
        df[f'{col}_roll30d'] = df[col].rolling(30, min_periods=15).mean()
        df[f'{col}_anomalia_60d'] = df[col].rolling(60, min_periods=30).mean() - 100.0

    df.bfill(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    out_path = os.path.join(PROC_DIR, 'dados_features.parquet')
    df.to_parquet(out_path, index=False)
    print(f"Features construídas e salvas em {out_path}")

if __name__ == "__main__":
    build_features()
