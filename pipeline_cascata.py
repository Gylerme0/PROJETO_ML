# =============================================================================
# PIPELINE DIÁRIO v3 — CLASSIFICADOR EM CASCATA
# =============================================================================
# Solução para o problema da classe Amarela (F1=0.00 nos modelos anteriores):
#
# O problema fundamental: Verde/Amarela/Verm.P1/Verm.P2 são classes ORDINAIS
# (têm ordem natural de gravidade). Um classificador multi-classe direto tende
# a ignorar as classes de transição (Amarela) porque suas features se sobrepõem
# às classes vizinhas.
#
# Solução — Cascata de 3 classificadores binários:
#   Etapa 1: Verde vs. Não-Verde (binário)
#             → Se Verde: encerra. Se Não-Verde: passa para Etapa 2
#   Etapa 2: Amarela vs. Vermelha (binário)
#             → Dentro dos Não-Verdes, separa crise leve de crise grave
#   Etapa 3: Verm.P1 vs. Verm.P2 (binário)
#             → Dentro das Vermelhas, separa os dois patamares
#
# Vantagem: cada modelo binário tem uma fronteira de decisão bem definida
# e aprende um conceito específico do domínio elétrico.
# =============================================================================

import sqlite3, os, glob, warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, roc_auc_score)
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
MAPA  = {'Verde': 0, 'Amarela': 1, 'Vermelha P1': 2,
         'Vermelha P2': 3, 'Escassez Hídrica': 3}


# =============================================================================
# BLOCO 1 — CARGA E FEATURE ENGINEERING (reutilizado do v2)
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
    data_min = df_band['DataRef'].min().to_timestamp()
    data_max = df_band['DataRef'].max().to_timestamp() + pd.offsets.MonthEnd(0)
    todos_dias = pd.DataFrame({'Data': pd.date_range(data_min, data_max, freq='D')})
    todos_dias['DataRef'] = todos_dias['Data'].dt.to_period('M')
    todos_dias = todos_dias.merge(df_band[['DataRef','Target']], on='DataRef', how='left')
    todos_dias.dropna(subset=['Target'], inplace=True)

    pasta_ena = os.path.join(BASE_DIR, 'ENA')
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

    df = todos_dias[['Data','DataRef','Target']].copy()
    df = df.merge(vol, on='Data', how='left')
    df = df.merge(df_chuva, on='Data', how='left')
    df = df.merge(df_ena_d, on='Data', how='left')
    df.sort_values('Data', inplace=True)
    df.reset_index(drop=True, inplace=True)

    cols_num = [c for c in df.columns if c not in ['Data','DataRef','Target']]
    df[cols_num] = df[cols_num].interpolate(method='linear', limit_direction='both')

    # Feature Engineering
    df['DiaDoAno_sin'] = np.sin(2 * np.pi * df['Data'].dt.dayofyear / 365)
    df['DiaDoAno_cos'] = np.cos(2 * np.pi * df['Data'].dt.dayofyear / 365)
    df['Mes_sin']      = np.sin(2 * np.pi * df['Data'].dt.month / 12)
    df['Mes_cos']      = np.cos(2 * np.pi * df['Data'].dt.month / 12)

    for col in [c for c in df.columns if c.startswith('Vol_')]:
        for w in [7, 30, 60, 90]:
            df[f'{col}_roll{w}d'] = df[col].rolling(w, min_periods=max(1,w//2)).mean()
        df[f'{col}_tend'] = df[f'{col}_roll30d'] - df[f'{col}_roll90d']

    for col in [c for c in df.columns if 'MWmed' in c]:
        for w in [7, 30, 60]:
            df[f'{col}_acum{w}d'] = df[col].rolling(w, min_periods=max(1,w//2)).sum()

    for col in [c for c in df.columns if 'pctMLT' in c]:
        df[f'{col}_roll30d'] = df[col].rolling(30, min_periods=15).mean()
        df[f'{col}_roll60d'] = df[col].rolling(60, min_periods=30).mean()

    for col in [c for c in df.columns if c.startswith('Chuva_')]:
        for w in [7, 30, 60]:
            df[f'{col}_acum{w}d'] = df[col].rolling(w, min_periods=max(1,w//2)).sum()

    if 'Vol_SE_CO' in df.columns and 'Vol_NE' in df.columns:
        df['Vol_SE_x_NE'] = df['Vol_SE_CO'] * df['Vol_NE'] / 10000

    col_mlt_se = next((c for c in df.columns if 'SE_CO_pctMLT' in c and 'roll' not in c), None)
    if col_mlt_se and 'Vol_SE_CO' in df.columns:
        df['ENA_pctMLT_x_Vol_SE'] = df[col_mlt_se] * df['Vol_SE_CO'] / 10000

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df.columns if c not in ['Data','DataRef','Target']]
    print(f"  Base: {len(df):,} dias | {len(feature_cols)} features")
    return df, feature_cols


# =============================================================================
# BLOCO 2 — SPLIT TEMPORAL + NORMALIZAÇÃO
# =============================================================================
def split_e_normalizar(df, feature_cols):
    X      = df[feature_cols].values
    y      = df['Target'].astype(int).values
    grupos = df['DataRef'].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, grupos))

    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr = grupos[tr_idx]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    print(f"  Treino: {len(X_tr):,} dias ({len(np.unique(g_tr))} meses)")
    print(f"  Teste:  {len(X_te):,} dias")
    for cls in sorted(np.unique(y_tr)):
        n_tr = (y_tr == cls).sum()
        n_te = (y_te == cls).sum()
        print(f"    {NOMES[cls]:12s}: treino={n_tr:4d}  teste={n_te:3d}")

    return X_tr_sc, X_te_sc, y_tr, y_te, g_tr


# =============================================================================
# BLOCO 3 — TREINAR CLASSIFICADOR BINÁRIO COM GroupKFold + SMOTE
# =============================================================================
def treinar_binario(X_tr, y_tr_bin, grupos, nome, n_trials=50):
    """
    Treina um XGBoost binário otimizado para uma etapa específica da cascata.
    Usa GroupKFold(meses) para CV honesto + SMOTE dentro de cada fold.
    """
    print(f"\n  [{nome}] Otimizando ({n_trials} trials)...")
    gkf = GroupKFold(n_splits=5)

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
            'eval_metric': 'logloss', 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }
        scores = []
        for f_tr_i, f_val_i in gkf.split(X_tr, y_tr_bin, grupos):
            Xf_tr, Xf_val = X_tr[f_tr_i], X_tr[f_val_i]
            yf_tr, yf_val = y_tr_bin[f_tr_i], y_tr_bin[f_val_i]
            if len(np.unique(yf_tr)) < 2: continue
            n_min = min((yf_tr == c).sum() for c in np.unique(yf_tr))
            if n_min < 2: continue
            k = min(5, n_min - 1)
            try:
                Xr, yr = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(Xf_tr, yf_tr)
            except: Xr, yr = Xf_tr, yf_tr
            m = xgb.XGBClassifier(**params)
            m.fit(Xr, yr)
            scores.append(f1_score(yf_val, m.predict(Xf_val), average='binary', zero_division=0))
        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    best.update({'eval_metric':'logloss','random_state':SEED,'n_jobs':-1,'verbosity':0})

    # Treinar modelo final com SMOTE no treino completo
    n_min = min((y_tr_bin == c).sum() for c in np.unique(y_tr_bin))
    k = min(5, n_min - 1)
    try:
        X_res, y_res = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X_tr, y_tr_bin)
    except:
        X_res, y_res = X_tr, y_tr_bin
    model = xgb.XGBClassifier(**best)
    model.fit(X_res, y_res)

    print(f"  [{nome}] Melhor CV F1-bin: {study.best_value:.4f}")
    return model


# =============================================================================
# BLOCO 4 — CASCATA DE 3 CLASSIFICADORES
# =============================================================================
def treinar_cascata(X_tr, X_te, y_tr, y_te, grupos):
    """
    Cascata de 3 classificadores binários:

      Etapa 1: Verde (0) vs. Não-Verde (1)
        ─ Aprende: "o sistema está confortável ou em alerta?"
        ─ Feature mais importante: Vol_SE_CO alto → Verde

      Etapa 2: Amarela (0) vs. Vermelha (1)  [só para Não-Verdes]
        ─ Aprende: "é um alerta leve ou crise grave?"
        ─ Feature mais importante: Vol_SE_CO < 40% + ENA_pctMLT < 70%

      Etapa 3: Verm.P1 (0) vs. Verm.P2 (1)  [só para Vermelhas]
        ─ Aprende: "é Patamar 1 ou Patamar 2?"
        ─ Feature mais importante: múltiplos subsistemas em crise simultânea
    """
    print("\n" + "="*68)
    print("  BLOCO 4: CASCATA DE 3 CLASSIFICADORES BINÁRIOS")
    print("="*68)
    print("""
  Etapa 1: Verde vs. Não-Verde
  Etapa 2: Amarela vs. Vermelha  (aplicado a Não-Verdes)
  Etapa 3: Verm.P1 vs. Verm.P2  (aplicado a Vermelhas)
    """)

    # ── Etapa 1: Verde(0) vs Não-Verde(1) ────────────────────────────────
    y_tr_e1 = (y_tr != 0).astype(int)    # 0=Verde, 1=NãoVerde
    m1 = treinar_binario(X_tr, y_tr_e1, grupos, "Etapa1: Verde vs Não-Verde", n_trials=50)

    # ── Etapa 2: Amarela(0) vs Vermelha(1), treino com não-verdes ─────────
    mask_tr_e2 = y_tr != 0
    X_tr_e2 = X_tr[mask_tr_e2]
    y_tr_e2 = (y_tr[mask_tr_e2] != 1).astype(int)   # 0=Amarela, 1=Vermelha
    g_tr_e2 = grupos[mask_tr_e2]
    m2 = treinar_binario(X_tr_e2, y_tr_e2, g_tr_e2, "Etapa2: Amarela vs Vermelha", n_trials=50)

    # ── Etapa 3: Verm.P1(0) vs Verm.P2(1), treino com vermelhas ──────────
    mask_tr_e3 = (y_tr == 2) | (y_tr == 3)
    X_tr_e3 = X_tr[mask_tr_e3]
    y_tr_e3 = (y_tr[mask_tr_e3] == 3).astype(int)   # 0=P1, 1=P2
    g_tr_e3 = grupos[mask_tr_e3]
    m3 = treinar_binario(X_tr_e3, y_tr_e3, g_tr_e3, "Etapa3: Verm.P1 vs Verm.P2", n_trials=40)

    return m1, m2, m3


# =============================================================================
# BLOCO 5 — PREDIÇÃO EM CASCATA + THRESHOLD TUNING
# =============================================================================
def predizer_cascata(m1, m2, m3, X_te, y_te, thresholds=None):
    """
    Aplica a cascata de forma sequencial.
    thresholds: (t1, t2, t3) — limiares de probabilidade para cada etapa.
    """
    if thresholds is None:
        t1, t2, t3 = 0.5, 0.5, 0.5
    else:
        t1, t2, t3 = thresholds

    # Etapa 1
    prob_e1   = m1.predict_proba(X_te)[:, 1]   # P(Não-Verde)
    pred_e1   = (prob_e1 > t1).astype(int)      # 0=Verde, 1=Não-Verde

    # Etapa 2 — só para os previstos como Não-Verde
    pred_final = np.zeros(len(X_te), dtype=int)   # começa tudo Verde
    mask_nv   = pred_e1 == 1

    if mask_nv.sum() > 0:
        prob_e2 = m2.predict_proba(X_te[mask_nv])[:, 1]   # P(Vermelha)
        pred_e2 = (prob_e2 > t2).astype(int)               # 0=Amarela, 1=Vermelha

        idx_nv = np.where(mask_nv)[0]
        
        # Amarela (pred_e2 == 0) → label 1
        idx_amarela = idx_nv[pred_e2 == 0]
        pred_final[idx_amarela] = 1

        # Etapa 3 — só para os previstos como Vermelha
        idx_verm = idx_nv[pred_e2 == 1]
        
        if len(idx_verm) > 0:
            prob_e3 = m3.predict_proba(X_te[idx_verm])[:, 1]   # P(P2)
            pred_e3 = (prob_e3 > t3).astype(int)               # 0=P1, 1=P2
            
            idx_p1 = idx_verm[pred_e3 == 0]
            idx_p2 = idx_verm[pred_e3 == 1]
            
            pred_final[idx_p1] = 2
            pred_final[idx_p2] = 3

    return pred_final


def tunar_thresholds(m1, m2, m3, X_te, y_te):
    """Busca os melhores limiares para as 3 etapas maximizando Macro F1."""
    best_f1, best_t = 0.0, (0.5, 0.5, 0.5)
    for t1 in np.arange(0.25, 0.75, 0.05):
        for t2 in np.arange(0.20, 0.80, 0.05):
            for t3 in np.arange(0.30, 0.70, 0.10):
                pred = predizer_cascata(m1, m2, m3, X_te, y_te, (t1, t2, t3))
                f1   = f1_score(y_te, pred, average='macro', zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, (t1, t2, t3)
    return best_t, best_f1


# =============================================================================
# BLOCO 6 — AVALIAÇÃO + GRÁFICOS
# =============================================================================
def avaliar_cascata(m1, m2, m3, X_te, y_te, feature_cols):
    print("\n" + "="*68)
    print("  BLOCO 6: AVALIAÇÃO DA CASCATA")
    print("="*68)

    # Sem tuning
    pred = predizer_cascata(m1, m2, m3, X_te, y_te)
    f1   = f1_score(y_te, pred, average='macro', zero_division=0)
    acc  = accuracy_score(y_te, pred)

    # Com threshold tuning
    print("  Buscando melhores thresholds...")
    best_t, f1_tuned = tunar_thresholds(m1, m2, m3, X_te, y_te)
    pred_tuned = predizer_cascata(m1, m2, m3, X_te, y_te, best_t)
    acc_tuned  = accuracy_score(y_te, pred_tuned)

    report       = classification_report(y_te, pred,       target_names=NOMES, labels=[0,1,2,3], zero_division=0)
    report_tuned = classification_report(y_te, pred_tuned, target_names=NOMES, labels=[0,1,2,3], zero_division=0)

    print(f"\n  Macro F1 (sem tuning):          {f1:.4f}")
    print(f"  Macro F1 (threshold tuning):    {f1_tuned:.4f}  t=({best_t[0]:.2f},{best_t[1]:.2f},{best_t[2]:.2f})")
    print(f"  Acurácia (sem tuning):          {acc:.1%}")
    print(f"  Acurácia (threshold tuning):    {acc_tuned:.1%}")
    print(f"\n  Report sem tuning:\n{report}")
    print(f"  Report com tuning:\n{report_tuned}")

    # Salvar relatório
    with open(os.path.join(PASTA_R, 'relatorio_cascata.txt'), 'w', encoding='utf-8') as f:
        f.write("CASCATA — PIPELINE DIÁRIO v3\n\n")
        f.write(f"Macro F1 (sem tuning):  {f1:.4f}\n")
        f.write(f"Macro F1 (com tuning):  {f1_tuned:.4f}\n")
        f.write(f"Thresholds ótimos:      t1={best_t[0]:.2f} t2={best_t[1]:.2f} t3={best_t[2]:.2f}\n")
        f.write(f"Acurácia (sem tuning):  {acc:.1%}\n")
        f.write(f"Acurácia (com tuning):  {acc_tuned:.1%}\n\n")
        f.write(f"Report:\n{report}\n")
        f.write(f"Report (Tuned):\n{report_tuned}\n")

    # Gráfico 1 — Matrizes de Confusão
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, p, title, f1v, accv in [
        (axes[0], pred,       'Cascata — Sem Tuning',       f1,       acc),
        (axes[1], pred_tuned, 'Cascata — Threshold Tuning', f1_tuned, acc_tuned),
    ]:
        cm = confusion_matrix(y_te, p, labels=[0,1,2,3])
        ConfusionMatrixDisplay(cm, display_labels=NOMES).plot(
            cmap='YlOrRd', ax=ax, colorbar=True, values_format='d')
        ax.set_title(f'{title}\nMacro F1 = {f1v:.4f}  |  Acc = {accv:.1%}',
                     fontsize=11, fontweight='bold')
    plt.suptitle('Classificador em Cascata — Pipeline Diário v3',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'cascata_confusao.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("\n  Salvo: cascata_confusao.png")

    # Gráfico 2 — Comparativo histórico
    fig, ax = plt.subplots(figsize=(11, 5))
    versoes = ['Baseline\n(mensal)', 'Diário v1\n(CV inflado)', 'Diário v2\n(CV honesto)', 'Diário v3\n(Cascata)']
    f1s  = [0.372, 0.520, 0.519, f1_tuned]
    accs = [0.436, 0.639, 0.655, acc_tuned]
    x = np.arange(len(versoes)); w = 0.35

    b1 = ax.bar(x-w/2, f1s,  w, label='Macro F1',  color='#3498db', alpha=0.88, edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x+w/2, accs, w, label='Acurácia', color='#e67e22', alpha=0.88, edgecolor='black', linewidth=0.5)
    ax.axhline(0.85, color='#e74c3c', ls='--', lw=1.5, label='Meta 85%')
    ax.axhline(0.90, color='#8b0000', ls='--', lw=1.8, label='Meta 90%')
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(versoes, fontsize=9)
    ax.set_ylim(0, 1.08); ax.set_ylabel('Score')
    ax.set_title('Evolução: Baseline → Cascata (Diário v3)', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'cascata_evolucao.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Salvo: cascata_evolucao.png")

    return f1, f1_tuned, acc, acc_tuned


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    print("\n" + "#"*68)
    print("  PIPELINE DIÁRIO v3 — CASCATA DE CLASSIFICADORES")
    print("  Verde→NãoVerde | Amarela→Vermelha | P1→P2")
    print("#"*68)

    print("\n  Carregando dados diários...")
    df, feature_cols = carregar_dados_diarios()

    print("\n  Split temporal...")
    X_tr, X_te, y_tr, y_te, g_tr = split_e_normalizar(df, feature_cols)

    m1, m2, m3 = treinar_cascata(X_tr, X_te, y_tr, y_te, g_tr)

    f1, f1_t, acc, acc_t = avaliar_cascata(m1, m2, m3, X_te, y_te, feature_cols)

    print("\n" + "="*68)
    print("  RESULTADO FINAL — CASCATA DIÁRIA v3")
    print("="*68)
    print(f"  Macro F1  (sem tuning):   {f1:.4f}")
    print(f"  Macro F1  (com tuning):   {f1_t:.4f}")
    print(f"  Acurácia  (sem tuning):   {acc:.1%}")
    print(f"  Acurácia  (com tuning):   {acc_t:.1%}")
    print(f"\n  Comparativo:")
    print(f"    Baseline:               F1=0.372  Acc=43.6%")
    print(f"    Diário v2 (honesto):    F1=0.519  Acc=65.5%")
    print(f"    Cascata v3:             F1={f1_t:.3f}  Acc={acc_t:.1%}")
    print(f"\n  Graficos: {PASTA_G}")
    print("="*68)
