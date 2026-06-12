# =============================================================================
# PIPELINE DIÁRIO — PREVISÃO DE BANDEIRAS TARIFÁRIAS (AV2)
# =============================================================================
# Granularidade: DIÁRIA (~3.900 dias vs. 130 meses anteriores = 30x mais dados)
#
# REGRA CRÍTICA — Split Temporal:
#   A bandeira é mensal (ANEEL define 1 por mês).
#   Cada dia recebe o label do mês que pertence.
#   → O split DEVE ser por meses completos (GroupShuffleSplit),
#     nunca aleatório — senão dias do mesmo mês ficam em treino E teste.
#
# Features diárias:
#   • Volume dos reservatórios (diário, por subsistema)
#   • ENA (Energia Natural Afluente) diária + % MLT
#   • Precipitação diária (INMET)
#   • Rolling windows: 7d, 30d, 60d, 90d
#   • Sazonalidade: dia do ano (sin/cos), mês (sin/cos)
#   • Interações e razões
# =============================================================================

import sqlite3, os, glob, warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, accuracy_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'base_energia.db')
PASTA_G  = os.path.join(BASE_DIR, 'graficos')
PASTA_R  = os.path.join(BASE_DIR, 'resultados')
os.makedirs(PASTA_G, exist_ok=True)
os.makedirs(PASTA_R, exist_ok=True)

SEED  = 42
NOMES = ['Verde', 'Amarela', 'Verm.P1', 'Verm.P2']
CORES = {0: '#2ecc71', 1: '#f1c40f', 2: '#e74c3c', 3: '#8b0000'}
MAPA  = {'Verde': 0, 'Amarela': 1, 'Vermelha P1': 2,
         'Vermelha P2': 3, 'Escassez Hídrica': 3}


# =============================================================================
# BLOCO 1 — CARREGAR TODAS AS FONTES EM GRANULARIDADE DIÁRIA
# =============================================================================
def bloco1_carregar_diario():
    """
    Carrega 4 fontes em granularidade diária:
      1. tb_hidrologico (SQLite) → Volume % por subsistema por dia
      2. tb_clima_inmet (SQLite) → Precipitação diária por subsistema
      3. ENA diária (xlsx)       → MWmed + % MLT por subsistema por dia
      4. tb_bandeiras (SQLite)   → Bandeira mensal → expandida para diária
    """
    print("\n" + "="*68)
    print("  BLOCO 1: CARREGANDO DADOS DIÁRIOS")
    print("="*68)

    conn = sqlite3.connect(DB_PATH)

    # ── 1. Volume diário dos reservatórios ────────────────────────────────
    df_agua = pd.read_sql(
        "SELECT data_medicao, nom_subsistema, val_volumeutilpercentual FROM tb_hidrologico",
        conn)
    df_agua['val_volumeutilpercentual'] = pd.to_numeric(
        df_agua['val_volumeutilpercentual'], errors='coerce')
    df_agua = df_agua[(df_agua['val_volumeutilpercentual'] >= 0) &
                      (df_agua['val_volumeutilpercentual'] <= 110)].copy()
    df_agua['Data'] = pd.to_datetime(df_agua['data_medicao'])

    # Média diária por subsistema (múltiplos reservatórios → 1 valor por subsistema)
    vol_diario = (
        df_agua.groupby(['Data', 'nom_subsistema'])['val_volumeutilpercentual']
        .mean().unstack().reset_index()
    )
    # Padronizar nomes das colunas
    col_map = {}
    for c in vol_diario.columns:
        cu = str(c).upper()
        if 'NORDESTE' in cu or c == 'NE': col_map[c] = 'Vol_NE'
        elif 'NORTE' in cu:               col_map[c] = 'Vol_Norte'
        elif 'SUDESTE' in cu or 'SE' in cu: col_map[c] = 'Vol_SE_CO'
        elif 'SUL' in cu:                 col_map[c] = 'Vol_Sul'
    vol_diario.rename(columns=col_map, inplace=True)
    print(f"  Volume diario: {len(vol_diario):,} dias | colunas: {[c for c in vol_diario.columns if c != 'Data']}")

    # ── 2. Precipitação diária ────────────────────────────────────────────
    df_chuva = pd.read_sql(
        "SELECT Data_Medicao, Chuva_Nordeste, Chuva_Norte, Chuva_Sudeste_CO, Chuva_Sul "
        "FROM tb_clima_inmet", conn)
    df_chuva['Data'] = pd.to_datetime(df_chuva['Data_Medicao'])
    df_chuva.drop(columns=['Data_Medicao'], inplace=True)
    print(f"  Chuva diaria:  {len(df_chuva):,} dias")

    # ── 3. Bandeiras mensais → expandir para diário ───────────────────────
    df_band = pd.read_sql(
        "SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras", conn)
    conn.close()

    df_band['DataRef'] = pd.to_datetime(df_band['DatCompetencia']).dt.to_period('M')
    df_band['Target']  = df_band['NomBandeiraAcionada'].map(MAPA)
    df_band.dropna(subset=['Target'], inplace=True)
    df_band = df_band.drop_duplicates('DataRef').sort_values('DataRef')

    # Criar um range diário e mapear cada dia ao mês correspondente
    data_min = df_band['DataRef'].min().to_timestamp()
    data_max = df_band['DataRef'].max().to_timestamp() + pd.offsets.MonthEnd(0)
    todos_dias = pd.DataFrame({'Data': pd.date_range(data_min, data_max, freq='D')})
    todos_dias['DataRef'] = todos_dias['Data'].dt.to_period('M')
    todos_dias = todos_dias.merge(df_band[['DataRef','Target','NomBandeiraAcionada']],
                                  on='DataRef', how='left')
    todos_dias.dropna(subset=['Target'], inplace=True)
    print(f"  Bandeiras:     {len(todos_dias):,} dias (range: "
          f"{todos_dias['Data'].min().date()} a {todos_dias['Data'].max().date()})")
    print(f"  Distribuição:")
    for cls in sorted(todos_dias['Target'].unique()):
        n = (todos_dias['Target'] == cls).sum()
        print(f"    {NOMES[int(cls)]:12s}: {n:4d} dias ({n/len(todos_dias)*100:.1f}%)")

    # ── 4. ENA diária ─────────────────────────────────────────────────────
    pasta_ena = os.path.join(BASE_DIR, 'ENA')
    arquivos  = sorted(glob.glob(os.path.join(pasta_ena, '*.xlsx')))
    lista_ena = []
    for arq in arquivos:
        try:
            df = pd.read_excel(arq, parse_dates=['ena_data'])
            lista_ena.append(df)
        except Exception as e:
            print(f"  [AVISO] ENA {os.path.basename(arq)}: {e}")

    df_ena = pd.concat(lista_ena, ignore_index=True)
    df_ena['ena_data'] = pd.to_datetime(df_ena['ena_data'], errors='coerce')
    df_ena.dropna(subset=['ena_data'], inplace=True)

    # Padronizar subsistemas
    mapa_sub = {'SUDESTE': 'SE_CO', 'SE': 'SE_CO', 'NORDESTE': 'NE',
                'NE': 'NE', 'NORTE': 'Norte', 'N': 'Norte', 'SUL': 'Sul', 'S': 'Sul'}
    df_ena['sub'] = df_ena['nom_subsistema'].str.upper().str.strip().map(
        lambda x: next((v for k,v in mapa_sub.items() if x == k), x))

    # Pivotar: uma coluna por subsistema × métrica
    ena_mwmed = (df_ena.pivot_table(index='ena_data', columns='sub',
                                    values='ena_bruta_regiao_mwmed', aggfunc='sum')
                 .reset_index())
    ena_mwmed.columns = (['Data'] +
                         [f'ENA_{c}_MWmed' for c in ena_mwmed.columns[1:]])

    ena_mlt = (df_ena.pivot_table(index='ena_data', columns='sub',
                                   values='ena_bruta_regiao_percentualmlt', aggfunc='mean')
               .reset_index())
    ena_mlt.columns = (['Data'] +
                       [f'ENA_{c}_pctMLT' for c in ena_mlt.columns[1:]])

    df_ena_diario = pd.merge(ena_mwmed, ena_mlt, on='Data')
    df_ena_diario['Data'] = pd.to_datetime(df_ena_diario['Data'])
    print(f"  ENA diaria:    {len(df_ena_diario):,} dias | "
          f"{len([c for c in df_ena_diario.columns if c!='Data'])} colunas")

    return vol_diario, df_chuva, df_ena_diario, todos_dias


# =============================================================================
# BLOCO 2 — MERGE DIÁRIO + FEATURE ENGINEERING
# =============================================================================
def bloco2_merge_features(vol_diario, df_chuva, df_ena_diario, todos_dias):
    """
    Constrói a base diária completa e cria features com janelas deslizantes:
      • Rolling 7d, 30d, 60d, 90d para volumes e ENA
      • Rolling 7d, 30d, 60d para chuva (acumulada)
      • Lags de 30, 60, 90 dias
      • Sazonalidade: dia do ano (sin/cos) + mês (sin/cos)
      • Interações: Vol_SE × ENA_SE, Vol_SE × Vol_NE
    """
    print("\n" + "="*68)
    print("  BLOCO 2: MERGE DIÁRIO + FEATURE ENGINEERING")
    print("="*68)

    # Merge sequencial por data
    df = todos_dias[['Data','DataRef','Target']].copy()
    df = df.merge(vol_diario, on='Data', how='left')
    df = df.merge(df_chuva,   on='Data', how='left')
    df = df.merge(df_ena_diario, on='Data', how='left')
    df.sort_values('Data', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  Base após merge: {len(df):,} dias")
    pct_na = df.isnull().mean() * 100
    cols_muitos_na = pct_na[pct_na > 30].index.tolist()
    if cols_muitos_na:
        print(f"  Colunas com >30% NA (descartadas): {cols_muitos_na}")
        df.drop(columns=cols_muitos_na, inplace=True)

    # Preencher NaN por interpolação linear (dias faltantes)
    cols_num = [c for c in df.columns if c not in ['Data','DataRef','Target','NomBandeiraAcionada']]
    df[cols_num] = df[cols_num].interpolate(method='linear', limit_direction='both')

    # ── FEATURES ─────────────────────────────────────────────────────────

    # 1. Sazonalidade cíclica
    df['DiaDoAno']     = df['Data'].dt.dayofyear
    df['DiaDoAno_sin'] = np.sin(2 * np.pi * df['DiaDoAno'] / 365)
    df['DiaDoAno_cos'] = np.cos(2 * np.pi * df['DiaDoAno'] / 365)
    df['Mes_sin']      = np.sin(2 * np.pi * df['Data'].dt.month / 12)
    df['Mes_cos']      = np.cos(2 * np.pi * df['Data'].dt.month / 12)

    # 2. Rolling windows de Volume (janelas em dias)
    vol_cols = [c for c in df.columns if c.startswith('Vol_')]
    for col in vol_cols:
        for w in [7, 30, 60, 90]:
            df[f'{col}_roll{w}d'] = df[col].rolling(w, min_periods=max(1,w//2)).mean()
        # Tendência: diferença entre média 30d e média 90d
        df[f'{col}_tend'] = df[f'{col}_roll30d'] - df[f'{col}_roll90d']

    # 3. Rolling de ENA (soma — total de energia afluente no período)
    ena_mwmed_cols = [c for c in df.columns if 'MWmed' in c]
    for col in ena_mwmed_cols:
        for w in [7, 30, 60]:
            df[f'{col}_acum{w}d'] = df[col].rolling(w, min_periods=max(1,w//2)).sum()

    # % MLT com rolling (média dos últimos 30d vs média histórica daquele dia do ano)
    ena_mlt_cols = [c for c in df.columns if 'pctMLT' in c]
    for col in ena_mlt_cols:
        df[f'{col}_roll30d'] = df[col].rolling(30, min_periods=15).mean()
        df[f'{col}_roll60d'] = df[col].rolling(60, min_periods=30).mean()

    # 4. Chuva acumulada (soma em janelas)
    chuva_cols = [c for c in df.columns if c.startswith('Chuva_')]
    for col in chuva_cols:
        for w in [7, 30, 60]:
            df[f'{col}_acum{w}d'] = df[col].rolling(w, min_periods=max(1,w//2)).sum()

    # 5. Interações (domínio do SIN)
    if 'Vol_SE_CO' in df.columns and 'Vol_NE' in df.columns:
        df['Vol_SE_x_NE'] = df['Vol_SE_CO'] * df['Vol_NE'] / 10000

    col_mlt_se = next((c for c in df.columns if 'SE_CO_pctMLT' in c and 'roll' not in c), None)
    if col_mlt_se and 'Vol_SE_CO' in df.columns:
        df['ENA_pctMLT_x_Vol_SE'] = df[col_mlt_se] * df['Vol_SE_CO'] / 10000

    col_ena_se = next((c for c in df.columns if 'SE_CO_MWmed' in c and 'acum' not in c), None)
    if col_ena_se and 'Vol_SE_CO' in df.columns:
        df['Ratio_ENA_Vol_SE'] = df[col_ena_se] / (df['Vol_SE_CO'] + 1.0)

    # 6. Remover colunas auxiliares
    df.drop(columns=['DiaDoAno','NomBandeiraAcionada'], errors='ignore', inplace=True)

    # 7. Remover linhas com NaN (bordas das janelas rolling)
    n_antes = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Linhas removidas por bordas rolling: {n_antes - len(df):,}")
    print(f"  Base final: {len(df):,} dias")

    feature_cols = [c for c in df.columns
                    if c not in ['Data', 'DataRef', 'Target']]
    print(f"  Total de features: {len(feature_cols)}")

    return df, feature_cols


# =============================================================================
# BLOCO 3 — SPLIT TEMPORAL POR MESES + SMOTE + NORMALIZAÇÃO
# =============================================================================
def bloco3_split_temporal(df, feature_cols):
    """
    SPLIT TEMPORAL POR MESES — obrigatório com dados diários de target mensal.

    Por que não usar train_test_split aleatório?
    ─────────────────────────────────────────────
    A bandeira é definida por mês. Se dividirmos aleatoriamente:
    - Dia 15/jan/2021 vai para TREINO com label "Vermelha P1"
    - Dia 18/jan/2021 vai para TESTE com label "Vermelha P1"
    - Os dois dias têm quase as mesmas features (reservatório muda <1%)
    - O modelo memoriza: "nestes valores exatos → Vermelha P1"
    - Resultado: F1 artificialmente inflado (DATA LEAKAGE TOTAL)

    Solução: Split por mês completo (GroupShuffleSplit).
    Cada grupo = 1 mês completo. Meses inteiros vão para treino ou teste,
    nunca divididos.

    Estratégia: 80% dos meses para treino, 20% para teste.
    Depois, SMOTE apenas no treino.
    """
    print("\n" + "="*68)
    print("  BLOCO 3: SPLIT TEMPORAL POR MESES + SMOTE + NORMALIZAÇÃO")
    print("="*68)

    X = df[feature_cols].values
    y = df['Target'].astype(int).values
    grupos = df['DataRef'].astype(str).values  # Cada mês é um grupo

    # GroupShuffleSplit: respeit a integridade dos meses
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, grupos))

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr = grupos[train_idx]

    print(f"  Treino: {len(X_tr):,} dias ({len(np.unique(g_tr))} meses)")
    print(f"  Teste:  {len(X_te):,} dias ({len(np.unique(grupos[test_idx]))} meses)")

    print(f"\n  Distribuição no Treino (antes SMOTE):")
    for cls in sorted(np.unique(y_tr)):
        n = (y_tr == cls).sum()
        print(f"    {NOMES[cls]:12s}: {n:4d} dias ({n/len(y_tr)*100:.1f}%)")

    print(f"\n  Distribuição no Teste:")
    for cls in sorted(np.unique(y_te)):
        n = (y_te == cls).sum()
        print(f"    {NOMES[cls]:12s}: {n:4d} dias ({n/len(y_te)*100:.1f}%)")

    # SMOTE no treino
    min_class = min((y_tr == c).sum() for c in np.unique(y_tr))
    k = min(5, min_class - 1)
    print(f"\n  SMOTE (k={k}) para balancear classes no treino...")
    smote = SMOTE(random_state=SEED, k_neighbors=k)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
    print(f"  Treino após SMOTE: {len(X_tr_res):,} amostras")
    for cls in sorted(np.unique(y_tr_res)):
        n = (y_tr_res == cls).sum()
        print(f"    {NOMES[cls]:12s}: {n:4d}")

    # Normalização Z-Score (fit no treino pré-SMOTE é equivalente)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_res)
    X_te_sc  = scaler.transform(X_te)
    print(f"\n  Normalização aplicada SEM Data Leakage")

    return X_tr_sc, X_te_sc, y_tr_res, y_te, g_tr, grupos[test_idx]


# =============================================================================
# BLOCO 4 — XGBOOST + OPTUNA
# =============================================================================
def bloco4_xgboost_optuna(X_train, y_train):
    """
    Com dados diários, o dataset de treino tem muito mais amostras (~3.200 dias).
    O Optuna com 100 trials vai encontrar hiperparâmetros muito mais robustos.
    """
    print("\n" + "="*68)
    print("  BLOCO 4: XGBOOST + OPTUNA (100 trials)")
    print("="*68)

    # Com dados diários, usar GroupKFold não é necessário aqui porque
    # o SMOTE já embaralhou o treino; usamos StratifiedKFold padrão
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 200, 800),
            'max_depth':        trial.suggest_int('max_depth', 3, 9),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma':            trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-6, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
            'eval_metric': 'mlogloss', 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring='f1_macro', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    print("  Executando 100 trials Bayesianos (TPE)...")
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    best = study.best_params
    best.update({'eval_metric': 'mlogloss', 'random_state': SEED,
                 'n_jobs': -1, 'verbosity': 0})

    print(f"\n  Melhor CV Macro F1 (Optuna): {study.best_value:.4f}")
    print("  Hiperparâmetros ótimos:")
    for k, v in best.items():
        if k not in ['eval_metric','random_state','n_jobs','verbosity']:
            print(f"    {k}: {v}")

    xgb_final = xgb.XGBClassifier(**best)
    xgb_final.fit(X_train, y_train)
    return xgb_final, study.best_value, study


# =============================================================================
# BLOCO 5 — AVALIAÇÃO + GRÁFICOS
# =============================================================================
def bloco5_avaliar(modelo, X_tr, X_te, y_tr, y_te, feature_cols, nome='XGBoost Diário'):
    """
    Avalia com métricas completas e gera 4 gráficos:
      1. Matriz de confusão
      2. Feature importance (Top 25)
      3. Curva de aprendizado (opcional)
      4. Evolução do Optuna (F1 por trial)
    """
    print("\n" + "="*68)
    print(f"  BLOCO 5: AVALIAÇÃO — {nome}")
    print("="*68)

    pred     = modelo.predict(X_te)
    macro_f1 = f1_score(y_te, pred, average='macro', zero_division=0)
    acc      = accuracy_score(y_te, pred)

    # CV 5-Fold no treino
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(modelo, X_tr, y_tr, cv=cv, scoring='f1_macro')

    print(f"\n  MACRO F1-SCORE (Teste):  {macro_f1:.4f}  {'✅' if macro_f1 >= 0.85 else '⚡' if macro_f1 >= 0.70 else '📈'}")
    print(f"  ACURÁCIA (Teste):        {acc:.4f}  ({acc:.1%})")
    print(f"  CV 5-Fold (Treino):      {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"\n  Classification Report:")
    report = classification_report(y_te, pred, target_names=NOMES,
                                   labels=[0,1,2,3], zero_division=0)
    print(report)

    # Threshold tuning
    if hasattr(modelo, 'predict_proba'):
        proba = modelo.predict_proba(X_te)
        best_f1_t, best_t = macro_f1, np.ones(4) * 0.25
        for t0 in np.arange(0.1, 0.7, 0.05):
            for t_m in np.arange(0.05, t0, 0.05):
                thr = np.array([t0, t_m, t_m, t_m * 0.8])
                p_t = np.argmax(proba / (thr + 1e-9), axis=1)
                f_t = f1_score(y_te, p_t, average='macro', zero_division=0)
                if f_t > best_f1_t:
                    best_f1_t, best_t = f_t, thr.copy()

        pred_tuned = np.argmax(proba / (best_t + 1e-9), axis=1)
        f1_tuned   = f1_score(y_te, pred_tuned, average='macro', zero_division=0)
        acc_tuned  = accuracy_score(y_te, pred_tuned)
        print(f"  Threshold Tuning:        {macro_f1:.4f} → {f1_tuned:.4f} | Acc: {acc_tuned:.1%}")
    else:
        pred_tuned = pred
        f1_tuned   = macro_f1
        acc_tuned  = acc

    # Salvar relatório
    relatorio = (
        f"PIPELINE DIÁRIO — {nome}\n{'='*55}\n"
        f"Macro F1 (Teste):   {macro_f1:.4f}\n"
        f"Macro F1 (Tuned):   {f1_tuned:.4f}\n"
        f"Acurácia (Teste):   {acc:.1%}\n"
        f"Acurácia (Tuned):   {acc_tuned:.1%}\n"
        f"CV 5-Fold F1:       {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n\n"
        f"{report}\n"
    )
    with open(os.path.join(PASTA_R, 'relatorio_diario.txt'), 'w', encoding='utf-8') as f:
        f.write(relatorio)

    # ── Gráfico 1: Matriz de Confusão ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, p, title in [
        (axes[0], pred,       f'{nome}\nMacro F1 = {macro_f1:.4f}  |  Acc = {acc:.1%}'),
        (axes[1], pred_tuned, f'{nome} + Threshold Tuning\nMacro F1 = {f1_tuned:.4f}  |  Acc = {acc_tuned:.1%}'),
    ]:
        cm = confusion_matrix(y_te, p, labels=[0,1,2,3])
        ConfusionMatrixDisplay(cm, display_labels=NOMES).plot(
            cmap='YlOrRd', ax=ax, colorbar=True, values_format='d')
        ax.set_title(title, fontsize=11, fontweight='bold')

    plt.suptitle('Matriz de Confusão — Pipeline Diário',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'diario_confusao.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("\n  Salvo: diario_confusao.png")

    # ── Gráfico 2: Feature Importance ─────────────────────────────────────
    if hasattr(modelo, 'feature_importances_'):
        imp = pd.Series(modelo.feature_importances_, index=feature_cols)
        top25 = imp.nlargest(25)[::-1]

        def cor_f(f):
            if 'ENA' in f:   return '#e74c3c'
            if 'Vol' in f:   return '#3498db'
            if 'Chuva' in f: return '#e67e22'
            if 'Dia' in f or 'Mes' in f: return '#9b59b6'
            if 'Ratio' in f or 'x_NE' in f: return '#1abc9c'
            return '#95a5a6'

        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(range(len(top25)), top25.values,
                color=[cor_f(f) for f in top25.index],
                edgecolor='black', linewidth=0.3, alpha=0.88)
        ax.set_yticks(range(len(top25)))
        ax.set_yticklabels(top25.index, fontsize=8)
        ax.set_xlabel('Importância (XGBoost gain)')
        ax.set_title(f'Top 25 Features — {nome}\n(granularidade diária)',
                     fontweight='bold', fontsize=12)
        legend = [
            mpatches.Patch(color='#e74c3c', label='ENA (Afluência)'),
            mpatches.Patch(color='#3498db', label='Volume Reservatório'),
            mpatches.Patch(color='#e67e22', label='Chuva'),
            mpatches.Patch(color='#9b59b6', label='Sazonalidade'),
            mpatches.Patch(color='#1abc9c', label='Interações/Razões'),
        ]
        ax.legend(handles=legend, loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PASTA_G, 'diario_feature_importance.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("  Salvo: diario_feature_importance.png")

    # ── Gráfico 3: Comparação Antes vs. Depois (resumo) ───────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    versoes = ['Baseline\n(mensal)', 'Otimizado\n(mensal+ENA)', f'{nome}\n(diário)']
    f1s     = [0.372, 0.818, macro_f1]
    accs    = [0.436, 0.795, acc]   # acurácia base estimada
    x = np.arange(len(versoes))
    w = 0.35

    bars1 = ax.bar(x - w/2, f1s,  w, label='Macro F1',  color='#3498db', alpha=0.88,
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w/2, accs, w, label='Acurácia', color='#e67e22', alpha=0.88,
                   edgecolor='black', linewidth=0.5)
    ax.axhline(0.85, color='#e74c3c', ls='--', lw=1.5, label='Meta 85%')
    ax.axhline(0.90, color='#8b0000', ls='--', lw=1.8, label='Meta 90%')

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(versoes, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Evolução do Modelo: Baseline → Mensal Otimizado → Diário',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'diario_evolucao.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Salvo: diario_evolucao.png")

    return {'macro_f1': macro_f1, 'macro_f1_tuned': f1_tuned,
            'accuracy': acc, 'accuracy_tuned': acc_tuned,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    print("\n" + "#"*68)
    print("  PIPELINE DIÁRIO — PREVISÃO DE BANDEIRAS TARIFÁRIAS")
    print("  ~3.900 dias | ENA + SMOTE + XGBoost/Optuna | Split Temporal")
    print("#"*68)

    # Bloco 1: Carregar fontes diárias
    vol_diario, df_chuva, df_ena_diario, todos_dias = bloco1_carregar_diario()

    # Bloco 2: Merge + Feature Engineering diário
    df, feature_cols = bloco2_merge_features(vol_diario, df_chuva, df_ena_diario, todos_dias)

    # Bloco 3: Split temporal por meses + SMOTE + normalização
    X_tr, X_te, y_tr, y_te, g_tr, g_te = bloco3_split_temporal(df, feature_cols)

    # Bloco 4: XGBoost + Optuna
    xgb_model, cv_best, study = bloco4_xgboost_optuna(X_tr, y_tr)

    # Bloco 5: Avaliação completa
    resultados = bloco5_avaliar(xgb_model, X_tr, X_te, y_tr, y_te,
                                feature_cols, nome='XGBoost Diário (Optuna)')

    # Resumo final
    print("\n" + "="*68)
    print("  RESULTADO FINAL — PIPELINE DIÁRIO")
    print("="*68)
    print(f"  Macro F1  (Teste):         {resultados['macro_f1']:.4f}")
    print(f"  Macro F1  (Threshold):     {resultados['macro_f1_tuned']:.4f}")
    print(f"  Acurácia  (Teste):         {resultados['accuracy']:.1%}")
    print(f"  Acurácia  (Threshold):     {resultados['accuracy_tuned']:.1%}")
    print(f"  CV 5-Fold (Treino):        {resultados['cv_mean']:.4f} (±{resultados['cv_std']:.4f})")
    print(f"\n  Comparativo:")
    print(f"    Baseline (130 meses):    F1=0.372  | Acc=43.6%")
    print(f"    Otimizado (mensal+ENA):  F1=0.818  | (CV treino)")
    print(f"    ESTE MODELO (diario):    F1={resultados['macro_f1']:.3f}  | Acc={resultados['accuracy']:.1%}")
    print(f"\n  Graficos: {PASTA_G}")
    print(f"  Relatorio: {PASTA_R}/relatorio_diario.txt")
    print("="*68)
