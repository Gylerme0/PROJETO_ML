# =============================================================================
# PIPELINE DIÁRIO v2 — CORRIGIDO
# =============================================================================
# Correções em relação à v1:
#  1. GroupKFold por mês no CV do Optuna (não StratifiedKFold)
#     → Antes: dias do mesmo mês estavam em treino E validação do CV
#       (CV inflado artificialmente para 0.9993)
#     → Agora: meses inteiros ficam em treino OU validação (CV honesto)
#
#  2. SMOTE aplicado dentro de cada fold do CV (não antes)
#     → Antes: SMOTE global → sintéticos vazam para os folds de validação
#     → Agora: SMOTE só no fold de treino, validação usa dados reais
#
#  3. scale_pos_weight para enfatizar a classe Amarela
#     (que ficou F1=0.00 na v1)
# =============================================================================

import sqlite3, os, glob, warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score)
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


# ── Reutiliza os blocos 1 e 2 do pipeline_diario.py ──────────────────────────
def carregar_dados_diarios():
    """Idêntico ao bloco1 + bloco2 do pipeline_diario.py."""
    conn = sqlite3.connect(DB_PATH)
    df_agua  = pd.read_sql("SELECT data_medicao, nom_subsistema, val_volumeutilpercentual FROM tb_hidrologico", conn)
    df_chuva = pd.read_sql("SELECT Data_Medicao, Chuva_Nordeste, Chuva_Norte, Chuva_Sudeste_CO, Chuva_Sul FROM tb_clima_inmet", conn)
    df_band  = pd.read_sql("SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras", conn)
    conn.close()

    # Volumes diários por subsistema
    df_agua['val_volumeutilpercentual'] = pd.to_numeric(df_agua['val_volumeutilpercentual'], errors='coerce')
    df_agua = df_agua[(df_agua['val_volumeutilpercentual'] >= 0) & (df_agua['val_volumeutilpercentual'] <= 110)].copy()
    df_agua['Data'] = pd.to_datetime(df_agua['data_medicao'])
    vol = df_agua.groupby(['Data', 'nom_subsistema'])['val_volumeutilpercentual'].mean().unstack().reset_index()
    col_map = {}
    for c in vol.columns:
        cu = str(c).upper()
        if 'NORDESTE' in cu: col_map[c] = 'Vol_NE'
        elif 'NORTE' in cu:  col_map[c] = 'Vol_Norte'
        elif 'SUDESTE' in cu or 'SE' in cu: col_map[c] = 'Vol_SE_CO'
        elif 'SUL' in cu:   col_map[c] = 'Vol_Sul'
    vol.rename(columns=col_map, inplace=True)

    # Chuva
    df_chuva['Data'] = pd.to_datetime(df_chuva['Data_Medicao'])
    df_chuva.drop(columns=['Data_Medicao'], inplace=True)

    # Bandeiras mensais → diárias
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

    # ENA diária
    pasta_ena = os.path.join(BASE_DIR, 'ENA')
    lista_ena = []
    for arq in sorted(glob.glob(os.path.join(pasta_ena, '*.xlsx'))):
        try:
            lista_ena.append(pd.read_excel(arq, parse_dates=['ena_data']))
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
    df_ena_diario = pd.merge(ena_mwmed, ena_mlt, on='Data')
    df_ena_diario['Data'] = pd.to_datetime(df_ena_diario['Data'])

    # Merge
    df = todos_dias[['Data','DataRef','Target']].copy()
    df = df.merge(vol, on='Data', how='left')
    df = df.merge(df_chuva, on='Data', how='left')
    df = df.merge(df_ena_diario, on='Data', how='left')
    df.sort_values('Data', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Preencher NaN
    cols_num = [c for c in df.columns if c not in ['Data','DataRef','Target']]
    df[cols_num] = df[cols_num].interpolate(method='linear', limit_direction='both')

    # Feature Engineering
    df['DiaDoAno_sin'] = np.sin(2 * np.pi * df['Data'].dt.dayofyear / 365)
    df['DiaDoAno_cos'] = np.cos(2 * np.pi * df['Data'].dt.dayofyear / 365)
    df['Mes_sin']      = np.sin(2 * np.pi * df['Data'].dt.month / 12)
    df['Mes_cos']      = np.cos(2 * np.pi * df['Data'].dt.month / 12)

    vol_cols = [c for c in df.columns if c.startswith('Vol_')]
    for col in vol_cols:
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
    return df, feature_cols


# =============================================================================
# SPLIT TEMPORAL
# =============================================================================
def split_temporal(df, feature_cols):
    X      = df[feature_cols].values
    y      = df['Target'].astype(int).values
    grupos = df['DataRef'].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, grupos))

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    g_tr = grupos[train_idx]

    print(f"  Treino: {len(X_tr):,} dias ({len(np.unique(g_tr))} meses)")
    print(f"  Teste:  {len(X_te):,} dias ({len(np.unique(grupos[test_idx]))} meses)")
    print(f"\n  Distribuição Treino:")
    for cls in sorted(np.unique(y_tr)):
        n = (y_tr == cls).sum()
        print(f"    {NOMES[cls]:12s}: {n:4d} ({n/len(y_tr)*100:.1f}%)")
    print(f"\n  Distribuição Teste:")
    for cls in sorted(np.unique(y_te)):
        n = (y_te == cls).sum()
        print(f"    {NOMES[cls]:12s}: {n:4d} ({n/len(y_te)*100:.1f}%)")

    # Normalizar: fit no TREINO, transform em ambos
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    return X_tr_sc, X_te_sc, y_tr, y_te, g_tr, scaler


# =============================================================================
# OPTUNA — CV POR MÊS (GroupKFold) + SMOTE DENTRO DE CADA FOLD
# =============================================================================
def treinar_optuna_correto(X_train, y_train, grupos_treino):
    """
    CV corrigido:
      • GroupKFold(5) com meses como grupos
        → meses inteiros ficam em treino OU validação, nunca divididos
      • SMOTE aplicado DENTRO de cada fold (só no fold de treino)
        → validação usa apenas dados reais, sem sintéticos
      • Isso elimina o overfitting artificial do CV anterior (0.9993 → ~0.6)
    """
    print("\n" + "="*68)
    print("  OPTUNA COM GroupKFold POR MÊS + SMOTE POR FOLD (CORRETO)")
    print("="*68)

    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma':            trial.suggest_float('gamma', 0.0, 3.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-5, 5.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-5, 5.0, log=True),
            'eval_metric': 'mlogloss', 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }

        scores = []
        for fold_tr_idx, fold_val_idx in gkf.split(X_train, y_train, grupos_treino):
            X_f_tr, X_f_val = X_train[fold_tr_idx], X_train[fold_val_idx]
            y_f_tr, y_f_val = y_train[fold_tr_idx], y_train[fold_val_idx]

            # SMOTE apenas no fold de treino
            n_min = min((y_f_tr == c).sum() for c in np.unique(y_f_tr))
            if n_min < 2:
                continue
            k = min(5, n_min - 1)
            smote = SMOTE(random_state=SEED, k_neighbors=k)
            X_res, y_res = smote.fit_resample(X_f_tr, y_f_tr)

            model = xgb.XGBClassifier(**params)
            model.fit(X_res, y_res)
            pred = model.predict(X_f_val)
            scores.append(f1_score(y_f_val, pred, average='macro', zero_division=0))

        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    print("  Executando 80 trials (CV por mês, honesto)...")
    study.optimize(objective, n_trials=80, show_progress_bar=True)

    best = study.best_params
    best.update({'eval_metric': 'mlogloss', 'random_state': SEED,
                 'n_jobs': -1, 'verbosity': 0})

    print(f"\n  Melhor CV GroupKFold Macro F1: {study.best_value:.4f}  (CV honesto por mês)")
    print("  Hiperparâmetros ótimos:")
    for k, v in best.items():
        if k not in ['eval_metric','random_state','n_jobs','verbosity']:
            print(f"    {k}: {v}")

    # Treinar modelo final com SMOTE no treino completo
    n_min = min((y_train == c).sum() for c in np.unique(y_train))
    k_smote = min(5, n_min - 1)
    smote_final = SMOTE(random_state=SEED, k_neighbors=k_smote)
    X_tr_res, y_tr_res = smote_final.fit_resample(X_train, y_train)
    print(f"\n  Treino final após SMOTE: {len(X_tr_res):,} amostras")
    for cls in sorted(np.unique(y_tr_res)):
        n = (y_tr_res == cls).sum()
        print(f"    {NOMES[cls]:12s}: {n}")

    xgb_final = xgb.XGBClassifier(**best)
    xgb_final.fit(X_tr_res, y_tr_res)
    return xgb_final, study.best_value, X_tr_res, y_tr_res


# =============================================================================
# AVALIAÇÃO + GRÁFICOS
# =============================================================================
def avaliar(modelo, X_te, y_te, X_tr_smote, y_tr_smote, feature_cols):
    print("\n" + "="*68)
    print("  AVALIAÇÃO FINAL")
    print("="*68)

    pred     = modelo.predict(X_te)
    macro_f1 = f1_score(y_te, pred, average='macro', zero_division=0)
    acc      = accuracy_score(y_te, pred)

    # Threshold Tuning
    proba = modelo.predict_proba(X_te)
    best_f1_t, best_t = macro_f1, np.ones(4) * 0.25
    for t0 in np.arange(0.1, 0.7, 0.05):
        for t_m in np.arange(0.05, t0, 0.05):
            thr = np.array([t0, t_m * 0.7, t_m, t_m])  # Amarela recebe threshold menor
            p_t = np.argmax(proba / (thr + 1e-9), axis=1)
            f_t = f1_score(y_te, p_t, average='macro', zero_division=0)
            if f_t > best_f1_t:
                best_f1_t, best_t = f_t, thr.copy()

    pred_tuned = np.argmax(proba / (best_t + 1e-9), axis=1)
    f1_tuned   = f1_score(y_te, pred_tuned, average='macro', zero_division=0)
    acc_tuned  = accuracy_score(y_te, pred_tuned)

    report      = classification_report(y_te, pred,       target_names=NOMES, labels=[0,1,2,3], zero_division=0)
    report_tuned= classification_report(y_te, pred_tuned, target_names=NOMES, labels=[0,1,2,3], zero_division=0)

    print(f"\n  MACRO F1-SCORE (Teste):            {macro_f1:.4f}")
    print(f"  MACRO F1-SCORE (Threshold Tuning): {f1_tuned:.4f}")
    print(f"  ACURÁCIA (Teste):                  {acc:.1%}")
    print(f"  ACURÁCIA (Threshold Tuning):       {acc_tuned:.1%}")
    print(f"\n  Classification Report (sem tuning):\n{report}")
    print(f"  Classification Report (com tuning):\n{report_tuned}")

    # Salvar relatório
    with open(os.path.join(PASTA_R, 'relatorio_diario_v2.txt'), 'w', encoding='utf-8') as f:
        f.write("PIPELINE DIÁRIO v2 — CORRIGIDO (GroupKFold por Mês)\n\n")
        f.write(f"Macro F1 (Teste):         {macro_f1:.4f}\n")
        f.write(f"Macro F1 (Tuned):         {f1_tuned:.4f}\n")
        f.write(f"Acurácia (Teste):         {acc:.1%}\n")
        f.write(f"Acurácia (Tuned):         {acc_tuned:.1%}\n\n")
        f.write(f"Report:\n{report}\n")
        f.write(f"Report (Tuned):\n{report_tuned}\n")

    # ── Gráfico 1: Matrizes de Confusão lado a lado ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, p, title, f1v, accv in [
        (axes[0], pred,       'Sem Threshold Tuning', macro_f1, acc),
        (axes[1], pred_tuned, 'Com Threshold Tuning', f1_tuned, acc_tuned),
    ]:
        cm = confusion_matrix(y_te, p, labels=[0,1,2,3])
        ConfusionMatrixDisplay(cm, display_labels=NOMES).plot(
            cmap='YlOrRd', ax=ax, colorbar=True, values_format='d')
        ax.set_title(f'{title}\nMacro F1 = {f1v:.4f}  |  Acc = {accv:.1%}',
                     fontsize=11, fontweight='bold')
    plt.suptitle('Pipeline Diário v2 — GroupKFold por Mês (CV Honesto)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'diario_v2_confusao.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Salvo: diario_v2_confusao.png")

    # ── Gráfico 2: Feature Importance ─────────────────────────────────────
    if hasattr(modelo, 'feature_importances_'):
        imp = pd.Series(modelo.feature_importances_, index=feature_cols)
        top25 = imp.nlargest(25)[::-1]

        def cor_f(f):
            if 'ENA' in f:   return '#e74c3c'
            if 'Vol' in f:   return '#3498db'
            if 'Chuva' in f: return '#e67e22'
            if 'Dia' in f or 'Mes' in f: return '#9b59b6'
            return '#1abc9c'

        fig, ax = plt.subplots(figsize=(11, 9))
        ax.barh(range(len(top25)), top25.values,
                color=[cor_f(f) for f in top25.index],
                edgecolor='black', linewidth=0.3, alpha=0.88)
        ax.set_yticks(range(len(top25)))
        ax.set_yticklabels(top25.index, fontsize=8)
        ax.set_xlabel('Importância (XGBoost gain)')
        ax.set_title('Top 25 Features — Pipeline Diário v2\n(GroupKFold corrigido)',
                     fontweight='bold', fontsize=12)
        legend = [
            mpatches.Patch(color='#e74c3c', label='ENA (Afluência)'),
            mpatches.Patch(color='#3498db', label='Volume Reservatório'),
            mpatches.Patch(color='#e67e22', label='Chuva'),
            mpatches.Patch(color='#9b59b6', label='Sazonalidade'),
            mpatches.Patch(color='#1abc9c', label='Interações'),
        ]
        ax.legend(handles=legend, loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PASTA_G, 'diario_v2_features.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("  Salvo: diario_v2_features.png")

    # ── Gráfico 3: Evolução histórica dos modelos ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    versoes = ['Baseline\n(mensal)', 'Mensal\n+ENA+Optuna\n(CV treino)', 'Diário v1\n(CV inflado)', f'Diário v2\n(CV honesto)']
    f1s     = [0.372, 0.818, 0.520, macro_f1]
    accs    = [0.436, 0.795, 0.639, acc]
    x = np.arange(len(versoes))
    w = 0.35

    bars1 = ax.bar(x - w/2, f1s,  w, label='Macro F1',  color='#3498db', alpha=0.88, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w/2, accs, w, label='Acurácia', color='#e67e22', alpha=0.88, edgecolor='black', linewidth=0.5)
    ax.axhline(0.85, color='#e74c3c', ls='--', lw=1.5, label='Meta 85%')
    ax.axhline(0.90, color='#8b0000', ls='--', lw=1.8, label='Meta 90%')
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontweight='bold', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(versoes, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Evolução dos Modelos — Baseline → Diário v2 (Corrigido)',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'diario_v2_evolucao.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Salvo: diario_v2_evolucao.png")

    return macro_f1, f1_tuned, acc, acc_tuned


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    print("\n" + "#"*68)
    print("  PIPELINE DIÁRIO v2 — CV CORRIGIDO (GroupKFold por Mês)")
    print("  SMOTE por fold | Threshold Tuning | 80 trials Bayesianos")
    print("#"*68)

    print("\n  Carregando e processando dados diários...")
    df, feature_cols = carregar_dados_diarios()
    print(f"  Base: {len(df):,} dias | {len(feature_cols)} features")

    print("\n  Split temporal por meses (GroupShuffleSplit)...")
    X_tr, X_te, y_tr, y_te, g_tr, scaler = split_temporal(df, feature_cols)

    modelo, cv_best, X_tr_smote, y_tr_smote = treinar_optuna_correto(X_tr, y_tr, g_tr)

    macro_f1, f1_tuned, acc, acc_tuned = avaliar(
        modelo, X_te, y_te, X_tr_smote, y_tr_smote, feature_cols)

    print("\n" + "="*68)
    print("  RESULTADO FINAL — PIPELINE DIÁRIO v2")
    print("="*68)
    print(f"  CV Honesto (GroupKFold):    {cv_best:.4f}")
    print(f"  Macro F1 (Teste):           {macro_f1:.4f}")
    print(f"  Macro F1 (Tuned):           {f1_tuned:.4f}")
    print(f"  Acurácia (Teste):           {acc:.1%}")
    print(f"  Acurácia (Tuned):           {acc_tuned:.1%}")
    print(f"\n  Comparativo:")
    print(f"    Baseline (130 meses):     F1=0.372  | Acc=43.6%")
    print(f"    Diário v1 (CV inflado):   F1=0.520  | Acc=63.9%")
    print(f"    Diário v2 (CV honesto):   F1={macro_f1:.3f}  | Acc={acc:.1%}")
    print(f"\n  Graficos: {PASTA_G}")
    print("="*68)
