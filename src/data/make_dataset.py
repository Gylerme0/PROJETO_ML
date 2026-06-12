import sqlite3
import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def make_dataset():
    print("Iniciando processamento de dados brutos...")
    db_path = os.path.join(RAW_DIR, 'base_energia.db')
    
    conn = sqlite3.connect(db_path)
    df_agua  = pd.read_sql("SELECT data_medicao, nom_subsistema, val_volumeutilpercentual FROM tb_hidrologico", conn)
    df_chuva = pd.read_sql("SELECT Data_Medicao, Chuva_Nordeste, Chuva_Norte, Chuva_Sudeste_CO, Chuva_Sul FROM tb_clima_inmet", conn)
    df_band  = pd.read_sql("SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras", conn)
    conn.close()

    df_agua['val_volumeutilpercentual'] = pd.to_numeric(df_agua['val_volumeutilpercentual'], errors='coerce')
    df_agua = df_agua[(df_agua['val_volumeutilpercentual'] >= 0) & (df_agua['val_volumeutilpercentual'] <= 110)].copy()
    df_agua['Data'] = pd.to_datetime(df_agua['data_medicao'])
    vol = df_agua.groupby(['Data','nom_subsistema'])['val_volumeutilpercentual'].mean().unstack().reset_index()
    
    col_map = {}
    for c in vol.columns:
        cu = str(c).upper()
        if 'NORDESTE' in cu: col_map[c] = 'Vol_NE'
        elif 'NORTE' in cu:  col_map[c] = 'Vol_Norte'
        elif 'SUDESTE' in cu or 'SE' in cu: col_map[c] = 'Vol_SE_CO'
        elif 'SUL' in cu:    col_map[c] = 'Vol_Sul'
    vol.rename(columns=col_map, inplace=True)

    df_chuva['Data'] = pd.to_datetime(df_chuva['Data_Medicao'])
    df_chuva.drop(columns=['Data_Medicao'], inplace=True)

    MAPA_BANDEIRAS = {'Verde': 0, 'Amarela': 1, 'Vermelha P1': 2, 'Vermelha P2': 3, 'Escassez Hídrica': 3}
    df_band['DataRef'] = pd.to_datetime(df_band['DatCompetencia']).dt.to_period('M')
    df_band['Target_Original'] = df_band['NomBandeiraAcionada'].map(MAPA_BANDEIRAS)
    df_band.dropna(subset=['Target_Original'], inplace=True)
    df_band = df_band.drop_duplicates('DataRef').sort_values('DataRef')
    
    df_band['Target_Financeiro'] = np.where(df_band['Target_Original'] >= 1, 1, 0)

    data_min = df_band['DataRef'].min().to_timestamp()
    data_max = df_band['DataRef'].max().to_timestamp() + pd.offsets.MonthEnd(0)
    todos_dias = pd.DataFrame({'Data': pd.date_range(data_min, data_max, freq='D')})
    todos_dias['DataRef'] = todos_dias['Data'].dt.to_period('M')
    todos_dias = todos_dias.merge(df_band[['DataRef','Target_Financeiro']], on='DataRef', how='left')
    todos_dias.dropna(subset=['Target_Financeiro'], inplace=True)

    pasta_ena = os.path.join(RAW_DIR, 'ENA')
    lista_ena = []
    for arq in sorted(glob.glob(os.path.join(pasta_ena, '*.xlsx'))):
        try: lista_ena.append(pd.read_excel(arq, parse_dates=['ena_data']))
        except: pass
    
    if lista_ena:
        df_ena = pd.concat(lista_ena, ignore_index=True)
        df_ena['ena_data'] = pd.to_datetime(df_ena['ena_data'], errors='coerce')
        df_ena.dropna(subset=['ena_data'], inplace=True)
        
        mapa_sub = {'SUDESTE':'SE_CO','SE':'SE_CO','NORDESTE':'NE','NE':'NE','NORTE':'Norte','N':'Norte','SUL':'Sul','S':'Sul'}
        df_ena['sub'] = df_ena['nom_subsistema'].str.upper().str.strip().map(lambda x: next((v for k,v in mapa_sub.items() if x==k), x))
        ena_mwmed = df_ena.pivot_table(index='ena_data', columns='sub', values='ena_bruta_regiao_mwmed', aggfunc='sum').reset_index()
        ena_mwmed.columns = ['Data'] + [f'ENA_{c}_MWmed' for c in ena_mwmed.columns[1:]]
        ena_mlt = df_ena.pivot_table(index='ena_data', columns='sub', values='ena_bruta_regiao_percentualmlt', aggfunc='mean').reset_index()
        ena_mlt.columns = ['Data'] + [f'ENA_{c}_pctMLT' for c in ena_mlt.columns[1:]]
        df_ena_d = pd.merge(ena_mwmed, ena_mlt, on='Data')
    else:
        df_ena_d = pd.DataFrame(columns=['Data'])

    df = todos_dias.copy()
    df = df.merge(vol, on='Data', how='left')
    df = df.merge(df_chuva, on='Data', how='left')
    if not df_ena_d.empty:
        df = df.merge(df_ena_d, on='Data', how='left')
        
    df.sort_values('Data', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['DataRef'] = df['DataRef'].astype(str) # Parquet compat

    out_path = os.path.join(PROC_DIR, 'dados_basicos.parquet')
    df.to_parquet(out_path, index=False)
    print(f"Dados salvos em {out_path}")

if __name__ == "__main__":
    make_dataset()
