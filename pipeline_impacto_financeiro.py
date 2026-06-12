# =============================================================================
# PROJETO ML - MODELO DE IMPACTO FINANCEIRO (ISENÇÃO VS SOBRETAXA)
# =============================================================================
# Objetivo: Prever o acionamento de Acréscimos Tarifários (Bandeira Amarela, 
# Vermelha P1 ou Vermelha P2) em oposição à Isenção Tarifária (Verde), 
# baseando-se em indicativos hidrológicos (ENA e Volume de Reservatórios).
#
# Acurácia de Validação Histórica Alvo: > 82.00%
# =============================================================================

import sqlite3, os, glob, warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from scipy.stats import mode

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)

# Caminhos principais do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'base_energia.db')
PASTA_G  = os.path.join(BASE_DIR, 'graficos')
PASTA_R  = os.path.join(BASE_DIR, 'resultados')
os.makedirs(PASTA_G, exist_ok=True)
os.makedirs(PASTA_R, exist_ok=True)

SEED  = 42
NOMES_FINANCEIROS = ['Isenção (Verde)', 'Sobretaxa (Amarela/Vermelha)']

# =============================================================================
# ETAPA 1 — INGESTÃO E FEATURE ENGINEERING (Engenharia de Recursos)
# =============================================================================
def carregar_e_processar_dados():
    """
    1. Lê os dados brutos do SQLite (Volume de Água e Bandeiras)
    2. Lê os arquivos Excel de ENA (Energia Natural Afluente)
    3. Cruza e constrói 'Features' (atrasos, anomalias e médias móveis)
    """
    print("[1/4] Extraindo dados do Banco SQLite e consolidando...")
    conn = sqlite3.connect(DB_PATH)
    df_agua  = pd.read_sql("SELECT data_medicao, nom_subsistema, val_volumeutilpercentual FROM tb_hidrologico", conn)
    df_chuva = pd.read_sql("SELECT Data_Medicao, Chuva_Nordeste, Chuva_Norte, Chuva_Sudeste_CO, Chuva_Sul FROM tb_clima_inmet", conn)
    df_band  = pd.read_sql("SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras", conn)
    conn.close()

    # Tratamento de Volume
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

    # Chuvas
    df_chuva['Data'] = pd.to_datetime(df_chuva['Data_Medicao'])
    df_chuva.drop(columns=['Data_Medicao'], inplace=True)

    # Bandeiras (Alvo do Modelo)
    MAPA_BANDEIRAS = {'Verde': 0, 'Amarela': 1, 'Vermelha P1': 2, 'Vermelha P2': 3, 'Escassez Hídrica': 3}
    df_band['DataRef'] = pd.to_datetime(df_band['DatCompetencia']).dt.to_period('M')
    df_band['Target_Original'] = df_band['NomBandeiraAcionada'].map(MAPA_BANDEIRAS)
    df_band.dropna(subset=['Target_Original'], inplace=True)
    df_band = df_band.drop_duplicates('DataRef').sort_values('DataRef')
    
    # ---------------------------------------------------------
    # DECISÃO DE NEGÓCIO: Agrupamento em Impacto Financeiro
    # 0 = Isenção (Bandeira Verde)
    # 1 = Sobretaxa (Amarela, P1 ou P2)
    # ---------------------------------------------------------
    df_band['Target_Financeiro'] = np.where(df_band['Target_Original'] >= 1, 1, 0)

    # Expansão Diária (para captar variações dia-a-dia de água vs Bandeira Mensal)
    data_min = df_band['DataRef'].min().to_timestamp()
    data_max = df_band['DataRef'].max().to_timestamp() + pd.offsets.MonthEnd(0)
    todos_dias = pd.DataFrame({'Data': pd.date_range(data_min, data_max, freq='D')})
    todos_dias['DataRef'] = todos_dias['Data'].dt.to_period('M')
    todos_dias = todos_dias.merge(df_band[['DataRef','Target_Financeiro']], on='DataRef', how='left')
    todos_dias.dropna(subset=['Target_Financeiro'], inplace=True)

    # Carga da ENA (Energia Natural Afluente)
    pasta_ena = os.path.join(BASE_DIR, 'data', 'ENA')
    lista_ena = []
    for arq in sorted(glob.glob(os.path.join(pasta_ena, '*.xlsx'))):
        try: lista_ena.append(pd.read_excel(arq, parse_dates=['ena_data']))
        except: pass
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

    # Mesclando Tudo
    df = todos_dias.copy()
    df = df.merge(vol, on='Data', how='left')
    df = df.merge(df_chuva, on='Data', how='left')
    df = df.merge(df_ena_d, on='Data', how='left')
    df.sort_values('Data', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Interpolação para tratar dias falhos de coleta
    cols_num = [c for c in df.columns if c not in ['Data','DataRef','Target_Financeiro']]
    df[cols_num] = df[cols_num].interpolate(method='linear', limit_direction='both')

    print("[2/4] Executando Engenharia de Variáveis (Sazonalidade e Anomalias)...")
    
    # 1. Indicadores Sazonais Cíclicos (Seno/Cosseno)
    df['DiaDoAno_sin'] = np.sin(2 * np.pi * df['Data'].dt.dayofyear / 365)
    df['DiaDoAno_cos'] = np.cos(2 * np.pi * df['Data'].dt.dayofyear / 365)
    df['MesHidrologico'] = ((df['Data'].dt.month - 10) % 12) + 1  
    df['MesHid_sin'] = np.sin(2 * np.pi * df['MesHidrologico'] / 12)
    df['MesHid_cos'] = np.cos(2 * np.pi * df['MesHidrologico'] / 12)

    # 2. Inércia Hidrológica (Rolling Windows / Médias Móveis)
    for col in [c for c in df.columns if c.startswith('Vol_')]:
        df[f'{col}_roll30d'] = df[col].rolling(30, min_periods=15).mean()
        df[f'{col}_roll90d'] = df[col].rolling(90, min_periods=45).mean()
        df[f'{col}_roll365d'] = df[col].rolling(365, min_periods=180).mean()
        # Anomalia: Quão pior estamos hoje comparado à média de 1 ano inteiro?
        df[f'{col}_anomalia_1ano'] = (df[col] - df[f'{col}_roll365d']) / (df[f'{col}_roll365d'] + 1e-6)

    for col in [c for c in df.columns if 'pctMLT' in c]:
        df[f'{col}_roll30d'] = df[col].rolling(30, min_periods=15).mean()
        # Anomalia: Quão abaixo dos 100% ideais da ENA estamos há 60 dias?
        df[f'{col}_anomalia_60d'] = df[col].rolling(60, min_periods=30).mean() - 100.0

    df.bfill(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df.columns if c not in ['Data','DataRef','Target_Financeiro']]
    return df, feature_cols


# =============================================================================
# ETAPA 2 — TREINAMENTO, VALIDAÇÃO CRUZADA E SUAVIZAÇÃO
# =============================================================================
def treinar_e_avaliar(df, feature_cols):
    """
    O XGBoost processa a série usando GroupKFold (agrupando dias no mesmo mês).
    Depois aplicamos 'Smooth Mensal' para refletir a política do ONS.
    """
    print(f"[3/4] Iniciando Treinamento XGBoost (Base de {len(df):,} dias | {len(feature_cols)} features)...")
    
    X = df[feature_cols].values
    y = df['Target_Financeiro'].values
    grupos = df['DataRef'].astype(str).values

    # Scaler Z-Score
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # GroupKFold garante que os dias de "Outubro/2018" não sejam quebrados entre Treino e Teste
    gkf = GroupKFold(n_splits=5)
    
    y_pred_diario = np.zeros(len(y))
    y_true_all = np.zeros(len(y))

    # XGBoost configurado com heurísticas robustas contra Overfitting
    clf = xgb.XGBClassifier(
        n_estimators=300, 
        max_depth=5, 
        learning_rate=0.05, 
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1, 
        random_state=SEED
    )

    for tr_i, te_i in gkf.split(X, y, grupos):
        X_tr, X_te = X[tr_i], X[te_i]
        y_tr, y_te = y[tr_i], y[te_i]
        
        clf.fit(X_tr, y_tr)
        y_pred_diario[te_i] = clf.predict(X_te)
        y_true_all[te_i] = y_te

    # ---------------------------------------------------------
    # SUAVIZAÇÃO MENSAL (Regra de Negócio)
    # Como a Bandeira é fechada para o mês inteiro, nós pegamos 
    # a predição majoritária do modelo (a Moda) dentro de cada mês.
    # ---------------------------------------------------------
    df['Pred_Diaria'] = y_pred_diario
    df['Pred_Suavizada'] = df.groupby('DataRef')['Pred_Diaria'].transform(lambda x: mode(x, keepdims=True).mode[0])
    y_pred_final = df['Pred_Suavizada'].values

    acc = accuracy_score(y, y_pred_final)
    print("\n" + "="*50)
    print(f"  ACURÁCIA FINAL (ISENÇÃO VS SOBRETAXA): {acc:.2%}")
    print("="*50)
    
    print("\nRelatório Científico de Métricas:")
    print(classification_report(y, y_pred_final, target_names=NOMES_FINANCEIROS, digits=3))

    return y, y_pred_final


# =============================================================================
# ETAPA 3 — GERAÇÃO DE GRÁFICOS (Matriz de Confusão)
# =============================================================================
def plotar_resultados(y_true, y_pred):
    print("[4/4] Gerando Matriz de Confusão de Impacto Financeiro...")
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=NOMES_FINANCEIROS).plot(
        cmap='Blues', ax=ax, colorbar=True, values_format='d'
    )
    
    ax.set_title(f'Validação de Impacto Financeiro\nAcurácia Global: {acc:.2%}', 
                 fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'resultado_financeiro_matriz.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  -> Gráfico salvo em: {os.path.join(PASTA_G, 'resultado_financeiro_matriz.png')}")

if __name__ == '__main__':
    df, features = carregar_e_processar_dados()
    y_t, y_p = treinar_e_avaliar(df, features)
    plotar_resultados(y_t, y_p)
    print("\nProcesso finalizado com Sucesso. O arquivo principal agora é o 'pipeline_impacto_financeiro.py'.")
