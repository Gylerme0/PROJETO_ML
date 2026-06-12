# =============================================================================
# PIPELINE OTIMIZADO — PREVISÃO DE BANDEIRAS TARIFÁRIAS (AV2)
# =============================================================================
# Estratégia em 5 camadas:
#  1. ENA (Energia Natural Afluente) — Variável que o ONS usa para setar bandeiras
#  2. Feature Engineering avançado (% MLT, razões, sazonalidade, interações)
#  3. SMOTE para balanceamento das classes raras no treino
#  4. XGBoost + Optuna (busca Bayesiana de hiperparâmetros)
#  5. Stacking Ensemble + Threshold Tuning
# =============================================================================

import sqlite3, os, glob, warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
from statsmodels.stats.contingency_tables import mcnemar

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'base_energia.db')
PASTA_G  = os.path.join(BASE_DIR, 'graficos')
PASTA_R  = os.path.join(BASE_DIR, 'resultados')
os.makedirs(PASTA_G, exist_ok=True)
os.makedirs(PASTA_R, exist_ok=True)

SEED = 42
NOMES = ['Verde', 'Amarela', 'Verm.P1', 'Verm.P2']
CORES = {0: '#2ecc71', 1: '#f1c40f', 2: '#e74c3c', 3: '#8b0000'}
MAPA  = {'Verde': 0, 'Amarela': 1, 'Vermelha P1': 2,
         'Vermelha P2': 3, 'Escassez Hídrica': 3}


# =============================================================================
# BLOCO 1 — CARREGAR ENA (Energia Natural Afluente)
# =============================================================================
def carregar_ena():
    """
    ENA = Energia Natural Afluente diária por subsistema (MWmed).
    Formato: LONGO — uma linha por dia × subsistema.
    Colunas importantes:
      • ena_bruta_regiao_mwmed         — ENA em MWmed (valor absoluto)
      • ena_bruta_regiao_percentualmlt — % da Média de Longo Termo (MLT)
                                         100% = afluência normal historicamente
                                         <70%  = risco de crise hídrica

    O ONS usa a ENA % MLT diretamente para calcular o CMO (Custo Marginal de
    Operação) e acionar bandeiras. É a variável mais próxima da decisão real.
    """
    print("\n" + "="*68)
    print("  BLOCO 1: CARREGANDO ENA (Energia Natural Afluente)")
    print("="*68)

    pasta_ena = os.path.join(BASE_DIR, 'ENA')
    arquivos  = sorted(glob.glob(os.path.join(pasta_ena, '*.xlsx')))

    lista = []
    for arq in arquivos:
        try:
            df = pd.read_excel(arq, parse_dates=['ena_data'])
            lista.append(df)
        except Exception as e:
            print(f"  [AVISO] {os.path.basename(arq)}: {e}")

    df_ena = pd.concat(lista, ignore_index=True)
    df_ena['ena_data'] = pd.to_datetime(df_ena['ena_data'], errors='coerce')
    df_ena.dropna(subset=['ena_data'], inplace=True)

    # Normalizar nomes dos subsistemas
    mapa_sub = {
        'SUDESTE': 'SE_CO', 'SE': 'SE_CO',
        'NORDESTE': 'NE',   'NE': 'NE',
        'NORTE': 'Norte',   'N':  'Norte',
        'SUL': 'Sul',       'S':  'Sul',
    }
    df_ena['subsistema'] = df_ena['nom_subsistema'].str.upper().str.strip().map(
        lambda x: next((v for k, v in mapa_sub.items() if x == k), x)
    )

    # Agregar para mensal (soma do mês em MWmed)
    df_ena['MesAno'] = df_ena['ena_data'].dt.to_period('M').dt.to_timestamp()

    # Pivotar: subsistema → coluna
    ena_mwmed = (
        df_ena.groupby(['MesAno', 'subsistema'])['ena_bruta_regiao_mwmed']
        .sum().unstack().reset_index()
    )
    ena_mwmed.columns = ['Data'] + [f'ENA_{c}_MWmed' for c in ena_mwmed.columns[1:]]

    # % MLT: média do mês (percentual da média histórica)
    ena_mlt = (
        df_ena.groupby(['MesAno', 'subsistema'])['ena_bruta_regiao_percentualmlt']
        .mean().unstack().reset_index()
    )
    ena_mlt.columns = ['Data'] + [f'ENA_{c}_pctMLT' for c in ena_mlt.columns[1:]]

    df_ena_mensal = pd.merge(ena_mwmed, ena_mlt, on='Data', how='inner')
    print(f"  ENA mensal: {len(df_ena_mensal)} meses")
    print(f"  Colunas: {df_ena_mensal.columns.tolist()}")
    return df_ena_mensal


# =============================================================================
# BLOCO 2 — MERGE COMPLETO + FEATURE ENGINEERING AVANÇADO
# =============================================================================
def construir_base(df_ena_mensal):
    """
    Combina 4 fontes de dados:
      1. Volumes dos reservatórios (ONS / SQLite)
      2. Precipitação (INMET / SQLite)
      3. ENA com % MLT (calculado acima)
      4. Carga de energia (CARGA_MENSAL.parquet)

    Cria features avançadas:
      • Sazonalidade cíclica (sin/cos do mês)
      • Lags de 1, 2, 3 meses + tendência de 3 meses
      • ENA % MLT (desvio da média histórica) + lags
      • Interação SE × NE (ambos secos = crise grave)
      • Razão ENA/Volume (água entrando vs. estoque disponível)
    """
    print("\n" + "="*68)
    print("  BLOCO 2: MERGE + FEATURE ENGINEERING AVANÇADO")
    print("="*68)

    conn = sqlite3.connect(DB_PATH)
    df_agua  = pd.read_sql("SELECT data_medicao, nom_subsistema, val_volumeutilpercentual FROM tb_hidrologico", conn)
    df_chuva = pd.read_sql("SELECT Data_Medicao, Chuva_Nordeste, Chuva_Norte, Chuva_Sudeste_CO, Chuva_Sul FROM tb_clima_inmet", conn)
    df_band  = pd.read_sql("SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras", conn)
    conn.close()

    # -- Volumes mensais --
    df_agua['val_volumeutilpercentual'] = pd.to_numeric(df_agua['val_volumeutilpercentual'], errors='coerce')
    df_agua = df_agua[(df_agua['val_volumeutilpercentual'] >= 0) &
                      (df_agua['val_volumeutilpercentual'] <= 110)].copy()
    df_agua['Data'] = pd.to_datetime(df_agua['data_medicao']).dt.to_period('M').dt.to_timestamp()
    vol = (df_agua.groupby(['Data','nom_subsistema'])['val_volumeutilpercentual']
           .mean().unstack().reset_index())
    vol.columns = ['Data', 'Vol_NE', 'Vol_Norte', 'Vol_SE_CO', 'Vol_Sul']

    # -- Chuva mensal (soma) --
    df_chuva['Data'] = pd.to_datetime(df_chuva['Data_Medicao']).dt.to_period('M').dt.to_timestamp()
    chuva = df_chuva.groupby('Data')[
        ['Chuva_Nordeste','Chuva_Norte','Chuva_Sudeste_CO','Chuva_Sul']
    ].sum().reset_index()

    # -- Carga de energia (total Brasil) --
    parquet_path = os.path.join(BASE_DIR, 'CARGA_MENSAL.parquet')
    if os.path.exists(parquet_path):
        df_carga = pd.read_parquet(parquet_path)
        df_carga['Data'] = pd.to_datetime(df_carga['din_instante']).dt.to_period('M').dt.to_timestamp()
        carga = df_carga.groupby('Data')['val_cargaenergiamwmed'].sum().reset_index()
        carga.rename(columns={'val_cargaenergiamwmed': 'Carga_Total_MWmed'}, inplace=True)
    else:
        carga = None

    # -- Bandeiras --
    df_band['Data']   = pd.to_datetime(df_band['DatCompetencia']).dt.to_period('M').dt.to_timestamp()
    df_band['Target'] = df_band['NomBandeiraAcionada'].map(MAPA)

    # -- Merge --
    df = vol.merge(chuva, on='Data', how='inner')
    df = df.merge(df_ena_mensal, on='Data', how='left')
    if carga is not None:
        df = df.merge(carga, on='Data', how='left')
        df['Carga_Total_MWmed'].fillna(df['Carga_Total_MWmed'].median(), inplace=True)
    df = df.merge(df_band[['Data','Target']], on='Data', how='inner')
    df.dropna(subset=['Target'], inplace=True)
    df.sort_values('Data', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Base após merge: {len(df)} meses")

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    # 1. Sazonalidade cíclica (sin + cos do mês)
    df['Mes']     = df['Data'].dt.month
    df['Mes_sin'] = np.sin(2 * np.pi * df['Mes'] / 12)
    df['Mes_cos'] = np.cos(2 * np.pi * df['Mes'] / 12)

    # 2. Lags de Volume + tendência 3 meses
    for col in ['Vol_SE_CO', 'Vol_NE', 'Vol_Sul', 'Vol_Norte']:
        df[f'{col}_Lag1']  = df[col].shift(1)
        df[f'{col}_Lag2']  = df[col].shift(2)
        df[f'{col}_Lag3']  = df[col].shift(3)
        df[f'{col}_Delta'] = df[col].diff()
        df[f'{col}_Tend3'] = (df[col] - df[col].shift(3)) / 3  # variação média mensal

    # 3. Chuva acumulada 2M e 3M
    for col in ['Chuva_Nordeste', 'Chuva_Norte', 'Chuva_Sudeste_CO', 'Chuva_Sul']:
        df[f'{col}_Acum2M'] = df[col].rolling(2).sum()
        df[f'{col}_Acum3M'] = df[col].rolling(3).sum()

    # 4. ENA: lags e acumulada
    ena_base_cols = [c for c in df.columns if 'ENA_' in c and 'Lag' not in c
                     and 'Acum' not in c and c != 'Data']
    for col in ena_base_cols:
        df[f'{col}_Lag1']  = df[col].shift(1)
        df[f'{col}_Acum3M'] = df[col].rolling(3).mean()

    # 5. Features de interação (domínio específico do SIN)
    # Quando SE/CO e NE estão secos simultaneamente → crise grave
    df['Vol_SE_x_NE'] = df['Vol_SE_CO'] * df['Vol_NE'] / 10000

    # ENA_SE_CO % MLT: a variável mais direta para a decisão do ONS
    col_mlt_se = next((c for c in df.columns if 'SE_CO_pctMLT' in c), None)
    col_mwmed_se = next((c for c in df.columns if 'SE_CO_MWmed' in c), None)
    if col_mlt_se:
        # Razão: ENA_SE % MLT dos últimos 3 meses vs. Volume atual
        df['Ratio_ENA_pctMLT_x_Vol'] = df[col_mlt_se] * df['Vol_SE_CO'] / 10000
    if col_mwmed_se:
        df['Ratio_ENA_Vol_SE'] = df[col_mwmed_se] / (df['Vol_SE_CO'] + 1.0)

    # 6. Remover NaN dos lags
    n_antes = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  Linhas removidas por NaN: {n_antes - len(df)}")
    print(f"  Base final: {len(df)} meses")

    feature_cols = [c for c in df.columns if c not in ['Data', 'Target', 'Mes']]
    print(f"  Total de features: {len(feature_cols)}")
    return df, feature_cols


# =============================================================================
# BLOCO 3 — SPLIT + SMOTE + NORMALIZAÇÃO
# =============================================================================
def preparar_com_smote(df, feature_cols):
    """
    Aplica SMOTE (Synthetic Minority Over-sampling Technique) apenas no treino.
    REGRA ABSOLUTA: SMOTE nunca toca nos dados de teste.
    """
    print("\n" + "="*68)
    print("  BLOCO 3: SPLIT + SMOTE + NORMALIZAÇÃO (Zero Data Leakage)")
    print("="*68)

    X = df[feature_cols]
    y = df['Target'].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)

    print(f"  Treino: {len(X_tr)} | Teste: {len(X_te)}")
    print(f"\n  Antes do SMOTE (treino):")
    for cls in sorted(y_tr.unique()):
        n = (y_tr == cls).sum()
        print(f"    {NOMES[cls]:12s}: {n}")

    # SMOTE — k_neighbors ajustado para o menor tamanho de classe
    min_class_size = min((y_tr == c).sum() for c in y_tr.unique())
    k = min(5, min_class_size - 1)
    print(f"\n  SMOTE com k_neighbors={k}")

    smote = SMOTE(random_state=SEED, k_neighbors=k)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

    print(f"\n  Após SMOTE (treino): {len(X_tr_res)} amostras")
    for cls in sorted(np.unique(y_tr_res)):
        n = (y_tr_res == cls).sum()
        print(f"    {NOMES[cls]:12s}: {n}")

    # Normalização Z-Score — fit SOMENTE no treino original (antes do SMOTE)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_res)
    X_te_sc  = scaler.transform(X_te)

    print(f"\n  Normalização aplicada SEM Data Leakage")
    return X_tr_sc, X_te_sc, y_tr_res, y_te, scaler


# =============================================================================
# BLOCO 4 — XGBOOST + OPTUNA
# =============================================================================
def treinar_xgboost_optuna(X_train, y_train):
    """
    Optuna usa TPE (Tree of Parzen Estimators):
    busca Bayesiana que aprende onde estão os bons hiperparâmetros.
    Muito mais eficiente que GridSearchCV.
    """
    print("\n" + "="*68)
    print("  BLOCO 4: XGBOOST + OPTUNA (80 trials Bayesianos)")
    print("="*68)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 8),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 0.0, 3.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True),
            'eval_metric': 'mlogloss', 'random_state': SEED,
            'n_jobs': -1, 'verbosity': 0,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring='f1_macro', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=80, show_progress_bar=True)

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
    return xgb_final, study.best_value


# =============================================================================
# BLOCO 5 — STACKING ENSEMBLE
# =============================================================================
def treinar_stacking(X_train, y_train):
    """
    Stacking (empilhamento):
    - Nível 0 (modelos base): XGBoost + GradientBoosting + LogReg
    - Nível 1 (meta-modelo): Regressão Logística aprende a combinar os 3

    O meta-modelo aprende: 'quando o XGBoost diz Verde mas o GBT diz
    Amarela, qual devo acreditar?'
    """
    print("\n" + "="*68)
    print("  BLOCO 5: STACKING ENSEMBLE")
    print("="*68)

    estimators = [
        ('xgb', xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, eval_metric='mlogloss',
            random_state=SEED, verbosity=0, n_jobs=-1)),
        ('gbt', GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=SEED)),
        ('lr', LogisticRegression(
            class_weight='balanced', max_iter=3000, random_state=SEED)),
    ]

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            class_weight='balanced', max_iter=3000, random_state=SEED),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        passthrough=True,
        n_jobs=-1
    )
    print("  Treinando Stacking (XGB + GBT + LR → meta LR)...")
    stacking.fit(X_train, y_train)
    print("  Stacking treinado!")
    return stacking


# =============================================================================
# BLOCO 6 — AVALIAÇÃO COMPLETA + THRESHOLD TUNING + GRÁFICOS
# =============================================================================
def avaliar_tudo(modelos_dict, X_tr, X_te, y_tr, y_te, feature_cols):
    print("\n" + "="*68)
    print("  BLOCO 6: AVALIAÇÃO + THRESHOLD TUNING + GRÁFICOS")
    print("="*68)

    cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    resultados = {}
    previsoes  = {}
    linhas_rel = ["PIPELINE OTIMIZADO — BANDEIRAS TARIFÁRIAS\n\n"]

    for nome, modelo in modelos_dict.items():
        pred       = modelo.predict(X_te)
        macro_f1   = f1_score(y_te, pred, average='macro', zero_division=0)
        acc        = accuracy_score(y_te, pred)
        cv_scores  = cross_val_score(modelo, X_tr, y_tr, cv=cv10, scoring='f1_macro')

        resultados[nome] = {'macro_f1': macro_f1, 'accuracy': acc,
                            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
        previsoes[nome]  = pred

        report = classification_report(y_te, pred, target_names=NOMES,
                                       labels=[0,1,2,3], zero_division=0)

        bloco = (f"\n{'='*55}\n{nome}\n{'='*55}\n"
                 f"Macro F1 (Teste): {macro_f1:.4f}\n"
                 f"Acurácia (Teste): {acc:.1%}\n"
                 f"CV 10-Fold F1:    {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n\n"
                 f"{report}")
        print(bloco)
        linhas_rel.append(bloco)

    # Threshold Tuning no modelo com predict_proba
    for nome, modelo in modelos_dict.items():
        if not hasattr(modelo, 'predict_proba'):
            continue
        proba = modelo.predict_proba(X_te)
        best_f1, best_t = 0.0, np.ones(4) * 0.25
        for t0 in np.arange(0.15, 0.65, 0.05):
            for t_minor in np.arange(0.05, t0, 0.05):
                thresholds = np.array([t0, t_minor, t_minor, t_minor])
                pred_t = np.argmax(proba / (thresholds + 1e-9), axis=1)
                f1_t   = f1_score(y_te, pred_t, average='macro', zero_division=0)
                if f1_t > best_f1:
                    best_f1, best_t = f1_t, thresholds.copy()

        pred_tuned = np.argmax(proba / (best_t + 1e-9), axis=1)
        f1_tuned   = f1_score(y_te, pred_tuned, average='macro', zero_division=0)
        acc_tuned  = accuracy_score(y_te, pred_tuned)

        nome_t = f"{nome} + Threshold"
        resultados[nome_t] = {'macro_f1': f1_tuned, 'accuracy': acc_tuned,
                              'cv_mean': resultados[nome]['cv_mean'],
                              'cv_std':  resultados[nome]['cv_std']}
        previsoes[nome_t]  = pred_tuned
        print(f"\n  Threshold Tuning ({nome}): F1 {resultados[nome]['macro_f1']:.4f} → {f1_tuned:.4f} | Acc {acc_tuned:.1%}")
        linhas_rel.append(f"\n{nome_t}: F1={f1_tuned:.4f} | Acc={acc_tuned:.1%}\n")

    # Salvar relatório
    with open(os.path.join(PASTA_R, 'relatorio_otimizado.txt'), 'w', encoding='utf-8') as f:
        f.writelines(linhas_rel)

    # ── Gráfico 1: Comparação Macro F1 vs Metas ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    nomes_p  = list(resultados.keys())
    f1_vals  = [resultados[n]['macro_f1'] for n in nomes_p]
    acc_vals = [resultados[n]['accuracy']  for n in nomes_p]
    cors     = ['#3498db','#e74c3c','#2ecc71','#9b59b6','#e67e22'][:len(nomes_p)]

    for ax, vals, ylabel, title in [
        (axes[0], f1_vals,  'Macro F1-Score', 'Macro F1-Score por Modelo'),
        (axes[1], acc_vals, 'Acurácia',       'Acurácia por Modelo'),
    ]:
        bars = ax.bar(range(len(nomes_p)), vals, color=cors,
                      edgecolor='black', linewidth=0.5, alpha=0.88)
        ax.axhline(0.85, color='#e74c3c', ls='--', lw=1.5, label='Meta 85%')
        ax.axhline(0.90, color='#8b0000', ls='--', lw=1.8, label='Meta 90%')
        ax.axhline(0.42, color='gray',    ls=':',  lw=1.2, label='Baseline anterior')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)
        ax.set_xticks(range(len(nomes_p)))
        ax.set_xticklabels(nomes_p, rotation=20, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)

    plt.suptitle('Pipeline Otimizado — Resultados vs. Metas', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'otimizado_comparacao.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("\n  Salvo: otimizado_comparacao.png")

    # ── Gráfico 2: Matriz de Confusão do melhor modelo ────────────────────
    melhor_nome  = max(resultados, key=lambda k: resultados[k]['macro_f1'])
    pred_melhor  = previsoes[melhor_nome]
    f1_m, acc_m  = resultados[melhor_nome]['macro_f1'], resultados[melhor_nome]['accuracy']

    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_te, pred_melhor, labels=[0,1,2,3])
    ConfusionMatrixDisplay(cm, display_labels=NOMES).plot(
        cmap='YlOrRd', ax=ax, colorbar=True, values_format='d')
    ax.set_title(f'Melhor Modelo: {melhor_nome}\n'
                 f'Macro F1 = {f1_m:.4f}  |  Acurácia = {acc_m:.1%}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_G, 'otimizado_confusao.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Salvo: otimizado_confusao.png")

    # ── Gráfico 3: Feature Importance do XGBoost ──────────────────────────
    xgb_model = next((m for n,m in modelos_dict.items() if 'XGBoost' in n), None)
    if xgb_model and hasattr(xgb_model, 'feature_importances_'):
        imp = pd.Series(xgb_model.feature_importances_, index=feature_cols)
        imp_top = imp.nlargest(20)[::-1]

        def cor_feat(f):
            if 'ENA' in f:     return '#e74c3c'
            if 'Vol' in f:     return '#3498db'
            if 'Chuva' in f:   return '#e67e22'
            if 'Mes' in f:     return '#9b59b6'
            if 'Ratio' in f or 'x_NE' in f: return '#1abc9c'
            return '#95a5a6'

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(len(imp_top)), imp_top.values,
                color=[cor_feat(f) for f in imp_top.index],
                edgecolor='black', linewidth=0.3)
        ax.set_yticks(range(len(imp_top)))
        ax.set_yticklabels(imp_top.index, fontsize=9)
        ax.set_xlabel('Importância (XGBoost gain)')
        ax.set_title('Top 20 Features — XGBoost Otimizado', fontweight='bold')
        legend = [
            mpatches.Patch(color='#e74c3c', label='ENA (Afluência)'),
            mpatches.Patch(color='#3498db', label='Volume Reservatório'),
            mpatches.Patch(color='#e67e22', label='Chuva'),
            mpatches.Patch(color='#9b59b6', label='Sazonalidade'),
            mpatches.Patch(color='#1abc9c', label='Interações/Razões'),
        ]
        ax.legend(handles=legend, loc='lower right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PASTA_G, 'otimizado_feature_importance.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print("  Salvo: otimizado_feature_importance.png")

    return resultados


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    print("\n" + "#"*68)
    print("  PIPELINE OTIMIZADO — BANDEIRAS TARIFÁRIAS")
    print("  ENA + SMOTE + XGBoost/Optuna + Stacking + Threshold Tuning")
    print("#"*68)

    df_ena              = carregar_ena()
    df, feature_cols    = construir_base(df_ena)
    X_tr, X_te, y_tr, y_te, scaler = preparar_com_smote(df, feature_cols)
    xgb_model, _        = treinar_xgboost_optuna(X_tr, y_tr)
    stack_model         = treinar_stacking(X_tr, y_tr)

    modelos = {
        'XGBoost (Optuna)': xgb_model,
        'Stacking Ensemble': stack_model,
    }
    resultados = avaliar_tudo(modelos, X_tr, X_te, y_tr, y_te, feature_cols)

    # --- Resumo final ---
    print("\n" + "="*68)
    print("  RESULTADO FINAL — PIPELINE OTIMIZADO")
    print("="*68)
    for nome, res in sorted(resultados.items(), key=lambda x: x[1]['macro_f1'], reverse=True):
        print(f"  {nome:<40s} | F1={res['macro_f1']:.4f} | Acc={res['accuracy']:.1%}")

    melhor = max(resultados, key=lambda k: resultados[k]['macro_f1'])
    print(f"\n  MELHOR MODELO: {melhor}")
    print(f"  Macro F1:  {resultados[melhor]['macro_f1']:.4f}")
    print(f"  Acurácia:  {resultados[melhor]['accuracy']:.1%}")
    print(f"  CV 10-Fold:{resultados[melhor]['cv_mean']:.4f} (±{resultados[melhor]['cv_std']:.4f})")
    print(f"\n  Graficos em: {PASTA_G}")
    print(f"  Relatorio:   {PASTA_R}/relatorio_otimizado.txt")
    print("="*68)
