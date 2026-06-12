# =============================================================================
# PIPELINE COMPLETO — PREVISÃO DE BANDEIRAS TARIFÁRIAS (AV2)
# =============================================================================
# Projeto Acadêmico: Machine Learning — AV2
# Aluno: Guilherme
#
# ARQUITETURA DO PIPELINE:
#   Bloco 1 → Conexão ao Banco SQLite e Extração dos Dados Brutos
#   Bloco 2 → Tratamento, Merge e Feature Engineering (Lags de Chuva)
#   Bloco 3 → Divisão Treino/Teste e Pré-processamento (Zero Data Leakage)
#   Bloco 4 → Treinamento: Regressão Logística + HistGradientBoosting
#   Bloco 5 → Avaliação: Classification Report, Macro F1, Validação Cruzada
#   Bloco 6 → Teste Estatístico de McNemar
#   Bloco 7 → Gráficos: Matriz de Confusão + Scatter Plot (Chuva vs Reservatório)
#
# DADOS REAIS UTILIZADOS:
#   • tb_hidrologico  → ONS   → Nível dos reservatórios (diário)
#   • tb_clima_inmet  → INMET → Precipitação diária (4 subsistemas)
#   • tb_bandeiras    → ANEEL → Bandeira acionada (mensal)
# =============================================================================

import sqlite3
import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para salvar gráficos sem janela
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings('ignore')  # Limpa warnings de convergência
sns.set_theme(style="whitegrid", font_scale=1.1)

# --- Caminhos e Constantes ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, 'base_energia.db')
PASTA_GRAFICOS   = os.path.join(BASE_DIR, 'graficos')
PASTA_RESULTADOS = os.path.join(BASE_DIR, 'resultados')
os.makedirs(PASTA_GRAFICOS, exist_ok=True)
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

SEED = 42  # Semente para reprodutibilidade

# Nomes e cores temáticas das bandeiras
NOMES_BANDEIRAS = ['Verde', 'Amarela', 'Verm. P1', 'Verm. P2']
CORES_BANDEIRAS = {0: '#2ecc71', 1: '#f1c40f', 2: '#e74c3c', 3: '#8b0000'}

# Mapeamento da variável-alvo
MAPA_BANDEIRAS = {
    'Verde': 0,
    'Amarela': 1,
    'Vermelha P1': 2,
    'Vermelha P2': 3,
    'Escassez Hídrica': 3  # Escassez = cenário extremo = equivalente a Verm. P2
}


# #############################################################################
# BLOCO 1 — CONEXÃO AO BANCO SQLITE E EXTRAÇÃO DOS DADOS BRUTOS
# #############################################################################
def bloco1_extracao():
    """
    Conecta ao SQLite (base_energia.db) e extrai as 3 tabelas:
      - tb_hidrologico  (ONS)   → Volume útil % dos reservatórios (diário)
      - tb_clima_inmet  (INMET) → Precipitação diária por subsistema
      - tb_bandeiras    (ANEEL) → Bandeira acionada (mensal)

    Retorna:
        df_agua (DataFrame):      Dados hidrológicos diários
        df_chuva (DataFrame):     Precipitação diária por subsistema
        df_bandeiras (DataFrame): Bandeiras mensais com Target numérico
    """
    print("\n" + "=" * 70)
    print("  BLOCO 1: EXTRAÇÃO DOS DADOS DO BANCO SQLite")
    print("=" * 70)

    conn = sqlite3.connect(DB_PATH)

    # --- 1.1 Dados Hidrológicos (ONS) ---
    df_agua = pd.read_sql(
        "SELECT data_medicao, nom_subsistema, val_volumeutilpercentual "
        "FROM tb_hidrologico",
        conn
    )
    print(f"  ✅ tb_hidrologico: {len(df_agua):,} registros diários")

    # --- 1.2 Dados Climáticos (INMET) ---
    df_chuva = pd.read_sql(
        "SELECT Data_Medicao, Chuva_Nordeste, Chuva_Norte, "
        "Chuva_Sudeste_CO, Chuva_Sul FROM tb_clima_inmet",
        conn
    )
    print(f"  ✅ tb_clima_inmet: {len(df_chuva):,} registros diários")

    # --- 1.3 Bandeiras Tarifárias (ANEEL) ---
    df_bandeiras = pd.read_sql(
        "SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras",
        conn
    )
    print(f"  ✅ tb_bandeiras:   {len(df_bandeiras)} registros mensais")

    conn.close()

    # Converter datas
    df_agua['data_medicao'] = pd.to_datetime(df_agua['data_medicao'])
    df_chuva['Data_Medicao'] = pd.to_datetime(df_chuva['Data_Medicao'])
    df_bandeiras['DatCompetencia'] = pd.to_datetime(df_bandeiras['DatCompetencia'])

    # Mapear bandeira → Target numérico
    df_bandeiras['Target'] = df_bandeiras['NomBandeiraAcionada'].map(MAPA_BANDEIRAS)
    df_bandeiras['Data'] = df_bandeiras['DatCompetencia'].dt.to_period('M').dt.to_timestamp()

    print(f"\n  📊 Distribuição da Variável-Alvo (Bandeiras):")
    nomes = {0: 'Verde', 1: 'Amarela', 2: 'Verm. P1', 3: 'Verm. P2/Escassez'}
    for k, v in df_bandeiras['Target'].value_counts().sort_index().items():
        pct = v / len(df_bandeiras) * 100
        print(f"      {nomes.get(k, k):20s}: {v:3d} meses ({pct:.1f}%)")

    return df_agua, df_chuva, df_bandeiras


# #############################################################################
# BLOCO 2 — TRATAMENTO, MERGE E FEATURE ENGINEERING
# #############################################################################
def bloco2_feature_engineering(df_agua, df_chuva, df_bandeiras):
    """
    1. Agrega dados diários para granularidade mensal
    2. Remove outliers físicos (reservatório < 0% ou > 110%)
    3. Cruza (merge) hidrologia + clima + bandeiras
    4. Cria Lagged Features de chuva acumulada (rolling window=2)
    5. Cria Lagged Features de volume dos reservatórios

    CONCEITO-CHAVE — INÉRCIA HÍDRICA:
        A chuva que cai em janeiro NÃO enche o reservatório instantaneamente.
        A água percorre a bacia hidrográfica ao longo de semanas/meses.
        Por isso, calculamos a chuva ACUMULADA dos últimos 2 meses
        (rolling(window=2).sum()) como feature preditiva.

    Retorna:
        df_final (DataFrame): Base consolidada pronta para modelagem
    """
    print("\n" + "=" * 70)
    print("  BLOCO 2: TRATAMENTO, MERGE E FEATURE ENGINEERING")
    print("=" * 70)

    # =========================================================================
    # 2.1 — REMOÇÃO DE OUTLIERS FÍSICOS (Reservatórios)
    # =========================================================================
    n_antes = len(df_agua)
    df_agua = df_agua[
        (df_agua['val_volumeutilpercentual'] >= 0) &
        (df_agua['val_volumeutilpercentual'] <= 110)
    ].copy()
    n_removidos = n_antes - len(df_agua)
    print(f"  🧹 Outliers removidos (Volume < 0% ou > 110%): {n_removidos:,} registros")

    # =========================================================================
    # 2.2 — AGREGAÇÃO MENSAL DOS RESERVATÓRIOS (diário → mensal)
    # =========================================================================
    df_agua['MesAno'] = df_agua['data_medicao'].dt.to_period('M').dt.to_timestamp()

    df_vol_mensal = (
        df_agua
        .groupby(['MesAno', 'nom_subsistema'])['val_volumeutilpercentual']
        .mean()
        .unstack()
        .reset_index()
    )
    df_vol_mensal.columns = ['Data', 'Vol_NE', 'Vol_Norte', 'Vol_SE_CO', 'Vol_Sul']
    print(f"  📦 Volumes mensais agregados: {len(df_vol_mensal)} meses")

    # =========================================================================
    # 2.3 — AGREGAÇÃO MENSAL DA PRECIPITAÇÃO (diário → mensal)
    # =========================================================================
    # Estratégia: SOMA mensal (total de chuva no mês, em mm)
    df_chuva['MesAno'] = df_chuva['Data_Medicao'].dt.to_period('M').dt.to_timestamp()

    df_chuva_mensal = (
        df_chuva
        .groupby('MesAno')[['Chuva_Nordeste', 'Chuva_Norte', 'Chuva_Sudeste_CO', 'Chuva_Sul']]
        .sum()
        .reset_index()
    )
    df_chuva_mensal.rename(columns={'MesAno': 'Data'}, inplace=True)
    print(f"  🌧️  Chuvas mensais agregadas: {len(df_chuva_mensal)} meses")

    # =========================================================================
    # 2.4 — MERGE: Hidrologia + Clima + Bandeiras
    # =========================================================================
    # Primeiro: volumes + chuva (ambos mensais)
    df_merged = pd.merge(df_vol_mensal, df_chuva_mensal, on='Data', how='inner')

    # Depois: resultado + bandeiras
    df_final = pd.merge(
        df_merged,
        df_bandeiras[['Data', 'Target']],
        on='Data',
        how='inner'
    )
    df_final.dropna(subset=['Target'], inplace=True)
    df_final.sort_values('Data', inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    print(f"  🔗 Base após merge: {len(df_final)} meses")

    # =========================================================================
    # 2.5 — FEATURE ENGINEERING: Lagged Features (Inércia Hídrica)
    # =========================================================================
    print("\n  ⚙️  Criando Lagged Features...")

    # Lags de Chuva Acumulada (rolling window=2 meses, soma)
    # Justificativa: a chuva acumulada nos últimos 2 meses é mais
    # representativa da situação hídrica do que a chuva de 1 mês isolado.
    colunas_chuva = ['Chuva_Nordeste', 'Chuva_Norte', 'Chuva_Sudeste_CO', 'Chuva_Sul']
    for col in colunas_chuva:
        df_final[f'{col}_Acum2M'] = df_final[col].rolling(window=2).sum()
        print(f"      ✅ {col}_Acum2M (soma chuva últimos 2 meses)")

    # Lags de Volume dos Reservatórios
    # Lag1 = mês anterior, Lag2 = 2 meses atrás
    # Delta = variação mensal (positiva = enchendo, negativa = secando)
    colunas_volume = ['Vol_SE_CO', 'Vol_NE', 'Vol_Sul', 'Vol_Norte']
    for col in colunas_volume:
        df_final[f'{col}_Lag1']  = df_final[col].shift(1)
        df_final[f'{col}_Lag2']  = df_final[col].shift(2)
        df_final[f'{col}_Delta'] = df_final[col].diff()
        print(f"      ✅ {col}_Lag1, {col}_Lag2, {col}_Delta")

    # Remover as primeiras linhas (que ficaram NaN por causa dos lags/rolling)
    n_antes_drop = len(df_final)
    df_final.dropna(inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    print(f"\n  🗑️  Linhas removidas por NaN dos Lags: {n_antes_drop - len(df_final)}")
    print(f"  ✅ Base final para ML: {len(df_final)} meses")

    # Listar todas as features criadas
    feature_cols = [c for c in df_final.columns if c not in ['Data', 'Target']]
    print(f"\n  📐 Total de Features: {len(feature_cols)}")
    for i, f in enumerate(feature_cols, 1):
        print(f"      {i:2d}. {f}")

    return df_final


# #############################################################################
# BLOCO 3 — DIVISÃO TREINO/TESTE E PRÉ-PROCESSAMENTO (ZERO DATA LEAKAGE)
# #############################################################################
def bloco3_preparacao(df_final):
    """
    1. Separa Features (X) e Target (y)
    2. Divide Treino/Teste com estratificação (70/30)
    3. Normaliza com StandardScaler — fit SOMENTE no treino

    REGRA CRÍTICA — ZERO DATA LEAKAGE:
        A normalização (StandardScaler) deve aprender a média e o desvio-padrão
        APENAS a partir dos dados de TREINO. Os dados de TESTE nunca devem
        influenciar o cálculo dessas estatísticas. Se fizermos isso errado,
        as informações do teste "vazam" para o treino e o modelo parece
        melhor do que realmente é — isso anula a validade do experimento.

    Retorna:
        X_train_scaled, X_test_scaled, y_train, y_test, colunas_features, df_final
    """
    print("\n" + "=" * 70)
    print("  BLOCO 3: DIVISÃO TREINO/TESTE (ZERO DATA LEAKAGE)")
    print("=" * 70)

    # Separar Features (X) e Target (y)
    colunas_features = [c for c in df_final.columns if c not in ['Data', 'Target']]
    X = df_final[colunas_features]
    y = df_final['Target'].astype(int)

    # --- DIVISÃO TREINO/TESTE (70/30) COM ESTRATIFICAÇÃO ---
    # stratify=y garante que a proporção de cada bandeira seja mantida
    # em ambos os conjuntos (essencial com desbalanceamento extremo)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    print(f"  📐 Features: {len(colunas_features)}")
    print(f"  📊 Amostras Treino: {len(X_train)} | Teste: {len(X_test)}")
    print(f"\n  Distribuição no Treino:")
    for cls in sorted(y_train.unique()):
        n = (y_train == cls).sum()
        print(f"      {NOMES_BANDEIRAS[cls]:10s}: {n} ({n/len(y_train)*100:.1f}%)")
    print(f"\n  Distribuição no Teste:")
    for cls in sorted(y_test.unique()):
        n = (y_test == cls).sum()
        print(f"      {NOMES_BANDEIRAS[cls]:10s}: {n} ({n/len(y_test)*100:.1f}%)")

    # --- NORMALIZAÇÃO Z-SCORE (ZERO DATA LEAKAGE) ---
    # 1. scaler.fit(X_train)     → aprende μ e σ somente do treino
    # 2. scaler.transform(X_train) → aplica transformação ao treino
    # 3. scaler.transform(X_test)  → aplica a MESMA transformação ao teste
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
    X_test_scaled  = scaler.transform(X_test)        # APENAS transform

    print(f"\n  ✅ StandardScaler aplicado SEM Data Leakage")
    print(f"     fit() executado SOMENTE nos {len(X_train)} registros de TREINO")

    return X_train_scaled, X_test_scaled, y_train, y_test, colunas_features, df_final


# #############################################################################
# BLOCO 4 — TREINAMENTO DOS DOIS MODELOS
# #############################################################################
def bloco4_treinamento(X_train, y_train):
    """
    Treina dois modelos para comparação:

    1. REGRESSÃO LOGÍSTICA (Baseline Interpretável):
       - Usa Softmax (multinomial) para classificar 4 bandeiras
       - class_weight='balanced' → penaliza erros em classes raras
       - Baseline: modelo simples para referência

    2. HistGradientBoostingClassifier (Estado da Arte):
       - Versão nativa do sklearn do Gradient Boosting (inspirado no LightGBM)
       - Suporta dados desbalanceados via sample_weight
       - Lida nativamente com NaN (mas já tratamos antes)
       - Referência: Ke et al. (2017) — LightGBM

    Retorna:
        modelos (dict): Dicionário com os dois modelos treinados
    """
    print("\n" + "=" * 70)
    print("  BLOCO 4: TREINAMENTO DOS MODELOS")
    print("=" * 70)

    modelos = {}

    # =========================================================================
    # Modelo 1: REGRESSÃO LOGÍSTICA (Baseline)
    # =========================================================================
    print("\n  🔵 Treinando Modelo 1: Regressão Logística (Multinomial)...")
    lr = LogisticRegression(
        class_weight='balanced',   # Compensa o desbalanceamento automaticamente
        max_iter=5000,             # Aumentado para garantir convergência
        solver='lbfgs',            # L-BFGS: eficiente para multiclasse (Softmax)
        random_state=SEED
    )
    lr.fit(X_train, y_train)
    modelos['Regressão Logística'] = lr
    print("     ✅ Treinado | Hiperparâmetros:")
    print("        class_weight='balanced', solver='lbfgs', multi_class='multinomial'")

    # =========================================================================
    # Modelo 2: HistGradientBoostingClassifier (Estado da Arte)
    # =========================================================================
    print("\n  🟢 Treinando Modelo 2: HistGradientBoostingClassifier...")

    # O HistGradientBoosting não aceita class_weight diretamente.
    # Solução: calcular sample_weight proporcional ao inverso da frequência.
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)

    hgb = HistGradientBoostingClassifier(
        max_iter=300,             # Número de árvores boosted
        max_depth=6,              # Limita profundidade contra overfitting
        learning_rate=0.05,       # Taxa de aprendizado conservadora
        min_samples_leaf=5,       # Mínimo de amostras por folha
        l2_regularization=1.0,    # Regularização L2 (Ridge)
        random_state=SEED
    )
    hgb.fit(X_train, y_train, sample_weight=sample_weights)
    modelos['HistGradientBoosting'] = hgb
    print("     ✅ Treinado | Hiperparâmetros:")
    print("        max_iter=300, max_depth=6, lr=0.05, l2_reg=1.0")
    print("        sample_weight='balanced' (via compute_sample_weight)")

    return modelos


# #############################################################################
# BLOCO 5 — AVALIAÇÃO: CLASSIFICATION REPORT, MACRO F1, VALIDAÇÃO CRUZADA
# #############################################################################
def bloco5_avaliacao(modelos, X_train, X_test, y_train, y_test):
    """
    Avalia os dois modelos com métricas rigorosas:
      - classification_report (Precision, Recall, F1 por classe)
      - Macro F1-Score (métrica principal — ignora acurácia global)
      - Validação Cruzada Estratificada 5-Fold no treino

    POR QUE MACRO F1 E NÃO ACURÁCIA?
        Com ~50% de bandeiras Verdes, um modelo que SEMPRE prevê "Verde"
        teria ~50% de acurácia, mas seria inútil clinicamente.
        O Macro F1-Score calcula o F1 de CADA classe e tira a MÉDIA,
        dando peso IGUAL para todas as bandeiras, inclusive as raras.

    Retorna:
        resultados (dict): Métricas de cada modelo
        previsoes (dict):  Previsões no teste (para McNemar)
    """
    print("\n" + "=" * 70)
    print("  BLOCO 5: AVALIAÇÃO DOS MODELOS")
    print("=" * 70)

    resultados = {}
    previsoes = {}

    # Estratégia de Validação Cruzada: Stratified 5-Fold
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for nome, modelo in modelos.items():
        print(f"\n  {'─' * 60}")
        print(f"  📊 {nome}")
        print(f"  {'─' * 60}")

        # --- Previsões no Teste ---
        pred = modelo.predict(X_test)
        previsoes[nome] = pred

        # --- Macro F1-Score (Métrica Principal) ---
        macro_f1 = f1_score(y_test, pred, average='macro', zero_division=0)

        # --- Classification Report ---
        report = classification_report(
            y_test, pred,
            target_names=NOMES_BANDEIRAS,
            labels=[0, 1, 2, 3],
            zero_division=0
        )

        # --- Validação Cruzada no TREINO (5-Fold) ---
        cv_scores = cross_val_score(
            modelo, X_train, y_train,
            cv=cv_strategy,
            scoring='f1_macro'
        )

        # Armazenar resultados
        resultados[nome] = {
            'macro_f1': macro_f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }

        # --- PRINTS DE SAÍDA ---
        print(f"\n  ★ MACRO F1-SCORE (Teste): {macro_f1:.4f}")
        print(f"\n  ★ VALIDAÇÃO CRUZADA (5-Fold no Treino):")
        print(f"    Folds: {['%.3f' % s for s in cv_scores]}")
        print(f"    Média: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
        print(f"\n  ★ CLASSIFICATION REPORT:")
        print(report)

    # --- Comparação Final ---
    print("\n  " + "=" * 60)
    print("  📊 COMPARAÇÃO FINAL DOS MODELOS")
    print("  " + "=" * 60)
    print(f"\n  {'Modelo':<30s} {'Macro F1 (Teste)':<20s} {'CV 5-Fold (Treino)'}")
    print(f"  {'─' * 70}")
    for nome, res in resultados.items():
        print(f"  {nome:<30s} {res['macro_f1']:<20.4f} {res['cv_mean']:.4f} (± {res['cv_std']:.4f})")

    melhor = max(resultados, key=lambda k: resultados[k]['macro_f1'])
    print(f"\n  🏆 MELHOR MODELO: {melhor} (Macro F1 = {resultados[melhor]['macro_f1']:.4f})")

    return resultados, previsoes


# #############################################################################
# BLOCO 6 — TESTE ESTATÍSTICO DE McNEMAR
# #############################################################################
def bloco6_mcnemar(previsoes, y_test, resultados):
    """
    Teste de McNemar: Compara se a diferença entre os dois modelos é
    estatisticamente significativa (α = 0.05).

    COMO FUNCIONA O McNEMAR:
        Monta uma tabela 2x2 de contingência:
        ┌──────────────────┬────────────────┬────────────────┐
        │                  │ Modelo B certo │ Modelo B errou │
        ├──────────────────┼────────────────┼────────────────┤
        │ Modelo A certo   │      a         │      b         │
        │ Modelo A errou   │      c         │      d         │
        └──────────────────┴────────────────┴────────────────┘

        Se b ≈ c (ambos erram/acertam igualmente), a diferença NÃO é significativa.
        Se b ≠ c (um acerta onde o outro erra), a diferença PODE ser significativa.

    Hipóteses:
        H₀: Os dois modelos têm desempenho equivalente
        H₁: Os modelos têm desempenho significativamente diferente
    """
    print("\n" + "=" * 70)
    print("  BLOCO 6: TESTE ESTATÍSTICO DE McNEMAR")
    print("=" * 70)

    nomes = list(previsoes.keys())
    pred_a = previsoes[nomes[0]]  # Regressão Logística
    pred_b = previsoes[nomes[1]]  # HistGradientBoosting

    # Montar tabela de contingência
    # a = ambos acertam, b = A acerta e B erra, c = A erra e B acerta, d = ambos erram
    a, b, c, d = 0, 0, 0, 0
    for yt, pa, pb in zip(y_test, pred_a, pred_b):
        a_certo = (pa == yt)
        b_certo = (pb == yt)
        if a_certo and b_certo:
            a += 1
        elif a_certo and not b_certo:
            b += 1
        elif not a_certo and b_certo:
            c += 1
        else:
            d += 1

    tabela_contingencia = [[a, b], [c, d]]

    print(f"\n  Comparando: {nomes[0]} vs {nomes[1]}")
    print(f"\n  Tabela de Contingência:")
    print(f"  ┌────────────────────────┬─────────────────┬─────────────────┐")
    print(f"  │                        │ {nomes[1]:15s} │ {nomes[1]:15s} │")
    print(f"  │                        │    Acertou      │     Errou       │")
    print(f"  ├────────────────────────┼─────────────────┼─────────────────┤")
    print(f"  │ {nomes[0]:22s} │                 │                 │")
    print(f"  │   Acertou              │     {a:3d}         │     {b:3d}         │")
    print(f"  │   Errou                │     {c:3d}         │     {d:3d}         │")
    print(f"  └────────────────────────┴─────────────────┴─────────────────┘")

    # Executar teste de McNemar (exato, adequado para amostras pequenas)
    resultado = mcnemar(tabela_contingencia, exact=True)
    p_value = resultado.pvalue

    print(f"\n  Estatística do Teste: {resultado.statistic:.4f}")
    print(f"  P-Value:              {p_value:.4f}")
    print(f"  Nível de significância: α = 0.05")

    if p_value < 0.05:
        print(f"\n  ✅ CONCLUSÃO: Diferença ESTATISTICAMENTE SIGNIFICATIVA (p < 0.05)")
        melhor = max(resultados, key=lambda k: resultados[k]['macro_f1'])
        print(f"     ➡️  O modelo '{melhor}' é comprovadamente superior.")
        print(f"     ➡️  Rejeitamos H₀: os modelos NÃO têm desempenho equivalente.")
    else:
        print(f"\n  ⚠️  CONCLUSÃO: Diferença NÃO estatisticamente significativa (p ≥ 0.05)")
        print(f"     ➡️  Não podemos afirmar que um modelo é superior ao outro.")
        print(f"     ➡️  Pela REGRA DE PARCIMÔNIA (Navalha de Occam), escolhemos o")
        print(f"        modelo mais simples: Regressão Logística.")

    return p_value


# #############################################################################
# BLOCO 7 — GRÁFICOS: MATRIZ DE CONFUSÃO + SCATTER PLOT
# #############################################################################
def bloco7_graficos(modelos, previsoes, resultados, y_test, df_final):
    """
    Gera os dois gráficos exigidos:

    1. MATRIZ DE CONFUSÃO do melhor modelo:
       Mostra onde o modelo acerta e erra para cada bandeira.

    2. SCATTER PLOT — Chuva Acumulada (X) vs Nível do Reservatório (Y):
       Colorido por bandeira, revela os clusters e fronteiras de decisão.
    """
    print("\n" + "=" * 70)
    print("  BLOCO 7: GERAÇÃO DE GRÁFICOS")
    print("=" * 70)

    melhor_nome = max(resultados, key=lambda k: resultados[k]['macro_f1'])
    pred_melhor = previsoes[melhor_nome]

    # =========================================================================
    # GRÁFICO 1: MATRIZ DE CONFUSÃO DO MELHOR MODELO
    # =========================================================================
    print(f"\n  📊 Gerando Matriz de Confusão ({melhor_nome})...")

    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, pred_melhor, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=NOMES_BANDEIRAS
    )
    disp.plot(
        cmap='YlOrRd',
        ax=ax,
        colorbar=True,
        values_format='d'
    )
    f1_val = resultados[melhor_nome]['macro_f1']
    ax.set_title(
        f'Matriz de Confusão — {melhor_nome}\n'
        f'Macro F1-Score = {f1_val:.4f}',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set_xlabel('Classe Prevista', fontsize=12)
    ax.set_ylabel('Classe Real', fontsize=12)
    plt.tight_layout()
    path_cm = os.path.join(PASTA_GRAFICOS, 'matriz_confusao_melhor_modelo.png')
    plt.savefig(path_cm, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"     ✅ Salvo: {path_cm}")

    # =========================================================================
    # GRÁFICO 2: SCATTER PLOT — CHUVA ACUMULADA vs NÍVEL DO RESERVATÓRIO
    # =========================================================================
    print(f"\n  📊 Gerando Scatter Plot (Chuva vs Reservatório)...")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Usando chuva acumulada do Sudeste/CO (eixo X) vs Volume do Sudeste/CO (eixo Y)
    # Justificativa: o subsistema Sudeste/CO concentra ~70% da capacidade
    # hidrelétrica brasileira, sendo o mais crítico para a decisão da bandeira.
    x_col = 'Chuva_Sudeste_CO_Acum2M'
    y_col = 'Vol_SE_CO'

    for target_val in sorted(df_final['Target'].unique()):
        subset = df_final[df_final['Target'] == target_val]
        target_int = int(target_val)
        ax.scatter(
            subset[x_col],
            subset[y_col],
            label=NOMES_BANDEIRAS[target_int],
            color=CORES_BANDEIRAS[target_int],
            s=80,
            edgecolors='black',
            linewidth=0.5,
            alpha=0.8,
            zorder=3
        )

    ax.set_xlabel('Chuva Acumulada — Sudeste/CO (mm, últimos 2 meses)', fontsize=12)
    ax.set_ylabel('Volume Útil do Reservatório — Sudeste/CO (%)', fontsize=12)
    ax.set_title(
        'Relação: Chuva Acumulada vs Nível do Reservatório\n'
        'Colorido por Bandeira Tarifária',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.legend(
        title='Bandeira',
        fontsize=11,
        title_fontsize=12,
        loc='upper left',
        framealpha=0.9
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path_scatter = os.path.join(PASTA_GRAFICOS, 'scatter_chuva_vs_reservatorio.png')
    plt.savefig(path_scatter, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"     ✅ Salvo: {path_scatter}")

    # =========================================================================
    # GRÁFICO BÔNUS: Comparação das Matrizes de Confusão (Lado a Lado)
    # =========================================================================
    print(f"\n  📊 Gerando comparação lado a lado das matrizes...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmaps_modelos = ['Blues', 'YlOrRd']

    for i, (nome, pred) in enumerate(previsoes.items()):
        cm_i = confusion_matrix(y_test, pred, labels=[0, 1, 2, 3])
        disp_i = ConfusionMatrixDisplay(
            confusion_matrix=cm_i,
            display_labels=NOMES_BANDEIRAS
        )
        disp_i.plot(cmap=cmaps_modelos[i], ax=axes[i], colorbar=False, values_format='d')
        f1_i = resultados[nome]['macro_f1']
        axes[i].set_title(f'{nome}\nMacro F1 = {f1_i:.4f}', fontsize=12, fontweight='bold')

    plt.suptitle(
        'Comparação: Matrizes de Confusão dos Dois Modelos',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    path_comp = os.path.join(PASTA_GRAFICOS, 'comparacao_matrizes_confusao.png')
    plt.savefig(path_comp, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"     ✅ Salvo: {path_comp}")

    print(f"\n  ✅ Todos os gráficos salvos em: {PASTA_GRAFICOS}")


# #############################################################################
# EXECUÇÃO PRINCIPAL
# #############################################################################
if __name__ == '__main__':
    print("\n" + "🔬" * 35)
    print("  PIPELINE COMPLETO — PREVISÃO DE BANDEIRAS TARIFÁRIAS (AV2)")
    print("  ETL → Feature Engineering → Modelagem → Avaliação → Gráficos")
    print("🔬" * 35)

    # Bloco 1: Extração dos dados
    df_agua, df_chuva, df_bandeiras = bloco1_extracao()

    # Bloco 2: Tratamento, Merge e Feature Engineering
    df_final = bloco2_feature_engineering(df_agua, df_chuva, df_bandeiras)

    # Bloco 3: Divisão Treino/Teste (Zero Data Leakage)
    X_train, X_test, y_train, y_test, colunas_features, df_final = bloco3_preparacao(df_final)

    # Bloco 4: Treinamento dos Modelos
    modelos = bloco4_treinamento(X_train, y_train)

    # Bloco 5: Avaliação (Classification Report, Macro F1, Validação Cruzada)
    resultados, previsoes = bloco5_avaliacao(modelos, X_train, X_test, y_train, y_test)

    # Bloco 6: Teste Estatístico de McNemar
    p_value = bloco6_mcnemar(previsoes, y_test, resultados)

    # Bloco 7: Gráficos (Matriz de Confusão + Scatter Plot)
    bloco7_graficos(modelos, previsoes, resultados, y_test, df_final)

    # --- RELATÓRIO FINAL RESUMIDO ---
    melhor = max(resultados, key=lambda k: resultados[k]['macro_f1'])
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETO FINALIZADO COM SUCESSO!")
    print("=" * 70)
    print(f"\n  🏆 Modelo Recomendado: {melhor}")
    print(f"     Macro F1-Score (Teste):  {resultados[melhor]['macro_f1']:.4f}")
    print(f"     CV 5-Fold (Treino):      {resultados[melhor]['cv_mean']:.4f} (± {resultados[melhor]['cv_std']:.4f})")
    print(f"     McNemar p-value:         {p_value:.4f}")
    print(f"\n  📁 Gráficos: {PASTA_GRAFICOS}")
    print(f"  📄 Banco:    {DB_PATH}")
    print("=" * 70)
