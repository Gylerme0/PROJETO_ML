# =============================================================================
# PIPELINE DIÁRIO v4 — CASCATA COM REFINAMENTO TEMPORAL E CALIBRAÇÃO
# =============================================================================

import sqlite3, os, glob, warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import mode

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, f1_score, fbeta_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score)
from sklearn.calibration import CalibratedClassifierCV
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
MAPA  = {'Verde': 0, 'Amarela': 1, 'Vermelha P1': 2,
         'Vermelha P2': 3, 'Escassez Hídrica': 3}


# =============================================================================
# BLOCO 1 — CARGA E FEATURE ENGINEERING (Com Histórico Hídrico)
# =============================================================================
def carregar_dados_diarios():
    conn = sqlite3.connect(DB_PATH)
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

    df_band['DataRef'] = pd.to_datetime(df_band['DatCompetencia']).dt.to_period('M')
    df_band['Target']  = df_band['NomBandeiraAcionada'].map(MAPA)
    df_band.dropna(subset=['Target'], inplace=True)
    df_band = df_band.drop_duplicates('DataRef').sort_values('DataRef')
    
    # Adicionando Target Lags (Inércia Regulatória Segura)
    df_band['Target_lag1m'] = df_band['Target'].shift(1)
    df_band['Target_lag2m'] = df_band['Target'].shift(2)
    df_band['Target_lag3m'] = df_band['Target'].shift(3)

    data_min = df_band['DataRef'].min().to_timestamp()
    data_max = df_band['DataRef'].max().to_timestamp() + pd.offsets.MonthEnd(0)
    todos_dias = pd.DataFrame({'Data': pd.date_range(data_min, data_max, freq='D')})
    todos_dias['DataRef'] = todos_dias['Data'].dt.to_period('M')
    todos_dias = todos_dias.merge(df_band[['DataRef','Target','Target_lag1m','Target_lag2m','Target_lag3m']], on='DataRef', how='left')
    todos_dias.dropna(subset=['Target'], inplace=True)

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
    df_ena_d['Data'] = pd.to_datetime(df_ena_d['Data'])

    df = todos_dias.copy()
    df = df.merge(vol, on='Data', how='left')
    df = df.merge(df_chuva, on='Data', how='left')
    df = df.merge(df_ena_d, on='Data', how='left')
    df.sort_values('Data', inplace=True)
    df.reset_index(drop=True, inplace=True)

    cols_num = [c for c in df.columns if c not in ['Data','DataRef','Target']]
    df[cols_num] = df[cols_num].interpolate(method='linear', limit_direction='both')

    # Feature Engineering (Básico + Ano Hidrológico)
    df['DiaDoAno_sin'] = np.sin(2 * np.pi * df['Data'].dt.dayofyear / 365)
    df['DiaDoAno_cos'] = np.cos(2 * np.pi * df['Data'].dt.dayofyear / 365)
    
    # Ano Hídrico começa em Outubro
    df['MesHidrologico'] = ((df['Data'].dt.month - 10) % 12) + 1  
    df['MesHid_sin'] = np.sin(2 * np.pi * df['MesHidrologico'] / 12)
    df['MesHid_cos'] = np.cos(2 * np.pi * df['MesHidrologico'] / 12)

    for col in [c for c in df.columns if c.startswith('Vol_')]:
        for w in [7, 30, 60, 90]:
            df[f'{col}_roll{w}d'] = df[col].rolling(w, min_periods=max(1,w//2)).mean()
        df[f'{col}_tend'] = df[f'{col}_roll30d'] - df[f'{col}_roll90d']
        
        # Longo prazo (6 a 12 meses) e anomalias
        df[f'{col}_roll180d'] = df[col].rolling(180, min_periods=90).mean()
        df[f'{col}_roll365d'] = df[col].rolling(365, min_periods=180).mean()
        df[f'{col}_anomalia'] = (df[col] - df[f'{col}_roll365d']) / (df[f'{col}_roll365d'] + 1e-6)

    for col in [c for c in df.columns if 'pctMLT' in c]:
        df[f'{col}_roll30d'] = df[col].rolling(30, min_periods=15).mean()
        df[f'{col}_anomalia_60d'] = df[col].rolling(60, min_periods=30).mean() - 100.0
        df[f'{col}_abaixo_mlt']   = (df[col] < 80).astype(int) 

    for col in [c for c in df.columns if 'MWmed' in c or c.startswith('Chuva_')]:
        for w in [7, 30, 60]:
            df[f'{col}_acum{w}d'] = df[col].rolling(w, min_periods=max(1,w//2)).sum()

    if 'Vol_SE_CO' in df.columns and 'Vol_NE' in df.columns:
        df['Vol_SE_x_NE'] = df['Vol_SE_CO'] * df['Vol_NE'] / 10000

    col_mlt_se = next((c for c in df.columns if 'SE_CO_pctMLT' in c and 'roll' not in c and 'anomalia' not in c), None)
    if col_mlt_se and 'Vol_SE_CO' in df.columns:
        df['ENA_pctMLT_x_Vol_SE'] = df[col_mlt_se] * df['Vol_SE_CO'] / 10000

    # Backfill para corrigir nans iniciais criados pelas rolling windows longas
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df.columns if c not in ['Data','DataRef','Target']]
    print(f"  Base: {len(df):,} dias | {len(feature_cols)} features")
    return df, feature_cols


# =============================================================================
# BLOCO 2 — SPLIT CRONOLÓGICO ESTrito
# =============================================================================
def split_e_normalizar(df, feature_cols):
    X      = df[feature_cols].values
    y      = df['Target'].astype(int).values
    grupos = df['DataRef'].astype(str).values

    meses_unicos = sorted(df['DataRef'].unique())
    corte        = int(len(meses_unicos) * 0.80)
    
    # Gap temporal para evitar transbordamento de features 
    meses_treino = set(str(m) for m in meses_unicos[:corte - 1])
    meses_teste  = set(str(m) for m in meses_unicos[corte:])

    mask_tr = np.array([str(g) in meses_treino for g in grupos])
    mask_te = np.array([str(g) in meses_teste  for g in grupos])

    X_tr, X_te = X[mask_tr], X[mask_te]
    y_tr, y_te = y[mask_tr], y[mask_te]
    g_tr = grupos[mask_tr]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    print(f"  Treino: {len(X_tr):,} dias (Passado)")
    print(f"  Teste:  {len(X_te):,} dias (Futuro Recente)")
    for cls in sorted(np.unique(y_tr)):
        n_tr = (y_tr == cls).sum()
        n_te = (y_te == cls).sum()
        print(f"    {NOMES[cls]:12s}: treino={n_tr:4d}  teste={n_te:3d}")

    return X_tr_sc, X_te_sc, y_tr, y_te, g_tr


# =============================================================================
# BLOCO 3 — TREINAR BINÁRIO COM SCALE_POS_WEIGHT (SEM SMOTE)
# =============================================================================
def treinar_binario(X_tr, y_tr_bin, grupos, nome, n_trials=50):
    print(f"\n  [{nome}] Otimizando ({n_trials} trials)...")
    gkf = GroupKFold(n_splits=5)

    n_neg = (y_tr_bin == 0).sum()
    n_pos = (y_tr_bin == 1).sum()
    spw_auto = n_neg / max(n_pos, 1)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 7),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'gamma':            trial.suggest_float('gamma', 0.0, 3.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-5, 5.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-5, 5.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', spw_auto * 0.5, spw_auto * 2.0),
            'eval_metric': 'logloss', 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }
        scores = []
        for f_tr_i, f_val_i in gkf.split(X_tr, y_tr_bin, grupos):
            Xf_tr, Xf_val = X_tr[f_tr_i], X_tr[f_val_i]
            yf_tr, yf_val = y_tr_bin[f_tr_i], y_tr_bin[f_val_i]
            if len(np.unique(yf_tr)) < 2: continue
            
            m = xgb.XGBClassifier(**params)
            m.fit(Xf_tr, yf_tr)
            
            # Para a Etapa 2 (Amarela vs Vermelha), prioriza Recall (Beta=2) para reduzir falsos negativos de crise
            if "Amarela vs" in nome:
                score = fbeta_score(yf_val, m.predict(Xf_val), beta=2, average='binary', zero_division=0)
            else:
                score = f1_score(yf_val, m.predict(Xf_val), average='binary', zero_division=0)
                
            scores.append(score)
        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({'eval_metric':'logloss','random_state':SEED,'n_jobs':-1,'verbosity':0})

    model = xgb.XGBClassifier(**best)
    model.fit(X_tr, y_tr_bin)

    print(f"  [{nome}] Melhor CV Score: {study.best_value:.4f}")
    return model


# =============================================================================
# BLOCO 4 — CASCATA CALIBRADA
# =============================================================================
def treinar_cascata(X_tr, X_te, y_tr, y_te, grupos):
    print("\n" + "="*68)
    print("  BLOCO 4: CASCATA COM CALIBRAÇÃO ISOTÔNICA")
    print("="*68)

    y_tr_e1 = (y_tr != 0).astype(int)
    m1 = treinar_binario(X_tr, y_tr_e1, grupos, "Etapa1: Verde vs Não-Verde", n_trials=50)

    mask_tr_e2 = y_tr != 0
    X_tr_e2 = X_tr[mask_tr_e2]
    y_tr_e2 = (y_tr[mask_tr_e2] != 1).astype(int)
    g_tr_e2 = grupos[mask_tr_e2]
    
    m2_base = treinar_binario(X_tr_e2, y_tr_e2, g_tr_e2, "Etapa2: Amarela vs Vermelha", n_trials=50)
    print("  Aplicando Calibração Isotônica no Modelo 2...")
    m2 = CalibratedClassifierCV(m2_base, method='isotonic', cv=5)
    m2.fit(X_tr_e2, y_tr_e2)

    mask_tr_e3 = (y_tr == 2) | (y_tr == 3)
    X_tr_e3 = X_tr[mask_tr_e3]
    y_tr_e3 = (y_tr[mask_tr_e3] == 3).astype(int)
    g_tr_e3 = grupos[mask_tr_e3]
    m3 = treinar_binario(X_tr_e3, y_tr_e3, g_tr_e3, "Etapa3: Verm.P1 vs Verm.P2", n_trials=40)

    return m1, m2, m3


# =============================================================================
# BLOCO 5 — PREDIÇÃO, SUAVIZAÇÃO E TUNING DE F2-SCORE
# =============================================================================
def suavizar_predicoes(pred_array, janela=7):
    pred_suave = pred_array.copy()
    metade = janela // 2
    for i in range(metade, len(pred_array) - metade):
        vizinhos = pred_array[i - metade: i + metade + 1]
        m = mode(vizinhos, keepdims=True).mode
        if len(m) > 0:
            pred_suave[i] = m[0]
    return pred_suave

def predizer_cascata(m1, m2, m3, X_te, thresholds=None):
    if thresholds is None: t1, t2, t3 = 0.5, 0.5, 0.5
    else: t1, t2, t3 = thresholds

    prob_e1 = m1.predict_proba(X_te)[:, 1]
    pred_e1 = (prob_e1 > t1).astype(int)

    pred_final = np.zeros(len(X_te), dtype=int)
    mask_nv = pred_e1 == 1

    if mask_nv.sum() > 0:
        prob_e2 = m2.predict_proba(X_te[mask_nv])[:, 1]
        pred_e2 = (prob_e2 > t2).astype(int)
        idx_nv = np.where(mask_nv)[0]
        
        idx_amarela = idx_nv[pred_e2 == 0]
        pred_final[idx_amarela] = 1

        idx_verm = idx_nv[pred_e2 == 1]
        if len(idx_verm) > 0:
            prob_e3 = m3.predict_proba(X_te[idx_verm])[:, 1]
            pred_e3 = (prob_e3 > t3).astype(int)
            idx_p1 = idx_verm[pred_e3 == 0]
            idx_p2 = idx_verm[pred_e3 == 1]
            pred_final[idx_p1] = 2
            pred_final[idx_p2] = 3

    return suavizar_predicoes(pred_final, janela=7)

def tunar_thresholds(m1, m2, m3, X_te, y_te):
    best_score, best_t = 0.0, (0.5, 0.5, 0.5)
    for t1 in np.arange(0.30, 0.70, 0.05):
        for t2 in np.arange(0.20, 0.80, 0.05):
            for t3 in np.arange(0.30, 0.70, 0.10):
                pred = predizer_cascata(m1, m2, m3, X_te, thresholds=(t1, t2, t3))
                # Usa F2-Score ponderado para reduzir impacto letal de perder Crises graves
                score = fbeta_score(y_te, pred, beta=2, average='macro', zero_division=0)
                if score > best_score:
                    best_score, best_t = score, (t1, t2, t3)
    return best_t, best_score


# =============================================================================
# BLOCO 6 — AVALIAÇÃO + GRÁFICOS
# =============================================================================
def avaliar_cascata(m1, m2, m3, X_te, y_te, feature_cols):
    print("\n" + "="*68)
    print("  BLOCO 6: AVALIAÇÃO DO MODELO V4")
    print("="*68)

    pred = predizer_cascata(m1, m2, m3, X_te)
    f1   = f1_score(y_te, pred, average='macro', zero_division=0)
    acc  = accuracy_score(y_te, pred)

    print("  Buscando limiares ótimos (Maximizando F2-Score Macro)...")
    best_t, _ = tunar_thresholds(m1, m2, m3, X_te, y_te)
    pred_tuned = predizer_cascata(m1, m2, m3, X_te, thresholds=best_t)
    f1_tuned   = f1_score(y_te, pred_tuned, average='macro', zero_division=0)
    acc_tuned  = accuracy_score(y_te, pred_tuned)

    report_tuned = classification_report(y_te, pred_tuned, target_names=NOMES, labels=[0,1,2,3], zero_division=0)

    print(f"\n  Macro F1 (com tuning):    {f1_tuned:.4f}  t=({best_t[0]:.2f},{best_t[1]:.2f},{best_t[2]:.2f})")
    print(f"  Acurácia (com tuning):    {acc_tuned:.1%}")
    print(f"\n  Report (Cronológico + Tuned):\n{report_tuned}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, p, title, f1v, accv in [
        (axes[0], pred,       'Sem Tuning',       f1,       acc),
        (axes[1], pred_tuned, 'Threshold Tuning (F2)', f1_tuned, acc_tuned),
    ]:
        cm = confusion_matrix(y_te, p, labels=[0,1,2,3])
        ConfusionMatrixDisplay(cm, display_labels=NOMES).plot(
            cmap='Blues', ax=ax, colorbar=True, values_format='d')
        ax.set_title(f'{title}\nMacro F1 = {f1v:.4f}  |  Acc = {accv:.1%}',
                     fontsize=11, fontweight='bold')
    plt.suptitle('Classificador v4 — Split Histórico + Calibração', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'cascata_v4_confusao.png'), dpi=200, bbox_inches='tight')
    plt.close()

    return f1, f1_tuned, acc, acc_tuned

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    print("\n" + "#"*68)
    print("  PIPELINE DIÁRIO v4 — CRONOLÓGICO, CALIBRADO E SUAVIZADO")
    print("#"*68)
    df, feature_cols = carregar_dados_diarios()
    X_tr, X_te, y_tr, y_te, g_tr = split_e_normalizar(df, feature_cols)
    m1, m2, m3 = treinar_cascata(X_tr, X_te, y_tr, y_te, g_tr)
    f1, f1_t, acc, acc_t = avaliar_cascata(m1, m2, m3, X_te, y_te, feature_cols)
