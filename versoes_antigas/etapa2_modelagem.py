# =============================================================================
# ETAPA 2 — PREPARAÇÃO, MODELAGEM E VALIDAÇÃO RIGOROSA
# =============================================================================
# Projeto: Previsão de Bandeiras Tarifárias com Machine Learning
#
# ESTE SCRIPT COBRE AS FASES 2, 3, 4 E 5 DA AVALIAÇÃO:
#   Fase 2 (Preparação): Features, Split, Normalização sem Data Leakage
#   Fase 3 (Modelagem - 20%): 3 algoritmos (LR, RF, SVM)
#   Fase 4 (Validação - 25%): CV, McNemar, Matrizes de Confusão, ROC
#   Fase 5 (Interpretação - 20%): Feature Importance, Insights
# =============================================================================

import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para salvar gráficos
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc)
from statsmodels.stats.contingency_tables import mcnemar

sns.set_theme(style="whitegrid", font_scale=1.05)
BASE_DIR = os.path.dirname(__file__)
PASTA_GRAFICOS = os.path.join(BASE_DIR, 'graficos')
PASTA_RESULTADOS = os.path.join(BASE_DIR, 'resultados')
os.makedirs(PASTA_GRAFICOS, exist_ok=True)
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

NOMES_BANDEIRAS = ['Verde', 'Amarela', 'Verm_P1', 'Verm_P2']
SEED = 42  # Semente fixa para reprodutibilidade (exigência AV2)


# =============================================================================
# FASE 2: PREPARAÇÃO DOS DADOS
# =============================================================================
def preparar_dados():
    """
    Extrai dados do SQLite, cria Lagged Features e divide Treino/Teste.
    
    CONCEITOS-CHAVE PARA ESTUDO:
    
    1. LAGGED FEATURES (Variáveis Defasadas):
       A água que cai como chuva hoje NÃO enche o reservatório amanhã.
       Existe uma "inércia hidrológica" — a água demora semanas/meses
       para percorrer a bacia até chegar ao reservatório. Por isso,
       criamos colunas com o valor do mês anterior (Lag1) e de 2 meses
       atrás (Lag2), para que o modelo aprenda essa dinâmica temporal.
    
    2. DATA LEAKAGE (Vazamento de Dados):
       NUNCA normalizar os dados ANTES de separar treino/teste!
       Se fizermos isso, as estatísticas do teste "vazam" para o treino
       e o modelo parece melhor do que realmente é. A regra é:
       - Calcular média/desvio SOMENTE no treino (fit)
       - Aplicar a mesma transformação no teste (transform)
    
    3. STRATIFY (Estratificação):
       Com dados desbalanceados (67 Verde vs 8 Escassez), se dividirmos
       aleatoriamente, pode acontecer de NENHUMA Vermelha P2 cair no teste.
       O stratify=y garante que a proporção se mantenha nas duas partes.
    """
    print("\n" + "=" * 70)
    print("  FASE 2: PREPARAÇÃO DOS DADOS")
    print("=" * 70)
    
    db_path = os.path.join(BASE_DIR, 'base_energia.db')
    conn = sqlite3.connect(db_path)
    
    # Extrair dados
    df_agua = pd.read_sql(
        "SELECT data_medicao, nom_subsistema, val_volumeutilpercentual FROM tb_hidrologico", conn)
    df_bandeiras = pd.read_sql(
        "SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras", conn)
    conn.close()
    
    # Agregar volumes diários em média mensal por subsistema
    df_agua['Data'] = pd.to_datetime(df_agua['data_medicao']).dt.to_period('M').dt.to_timestamp()
    df_agua_mensal = (
        df_agua.groupby(['Data', 'nom_subsistema'])['val_volumeutilpercentual']
        .mean().unstack().reset_index()
    )
    df_agua_mensal.columns = ['Data', 'Vol_NE', 'Vol_Norte', 'Vol_SE_CO', 'Vol_Sul']
    
    # Preparar Target
    df_bandeiras['Data'] = pd.to_datetime(df_bandeiras['DatCompetencia']).dt.to_period('M').dt.to_timestamp()
    mapa = {'Verde': 0, 'Amarela': 1, 'Vermelha P1': 2, 'Vermelha P2': 3, 'Escassez Hídrica': 3}
    df_bandeiras['Target'] = df_bandeiras['NomBandeiraAcionada'].map(mapa)
    
    # Merge
    df = pd.merge(df_agua_mensal, df_bandeiras[['Data', 'Target']], on='Data', how='inner')
    df.dropna(subset=['Target'], inplace=True)
    df.sort_values('Data', inplace=True)
    
    # --- Incorporar Carga de Energia (CARGA_MENSAL.parquet) ---
    parquet_path = os.path.join(BASE_DIR, 'CARGA_MENSAL.parquet')
    if os.path.exists(parquet_path):
        print("  📦 Incorporando Carga de Energia (MWmed) do ONS...")
        df_carga = pd.read_parquet(parquet_path)
        df_carga['Data'] = pd.to_datetime(df_carga['din_instante']).dt.to_period('M').dt.to_timestamp()
        # Somar a carga de todos os subsistemas = carga total Brasil
        carga_total = df_carga.groupby('Data')['val_cargaenergiamwmed'].sum().reset_index()
        carga_total.rename(columns={'val_cargaenergiamwmed': 'Carga_Total_MWmed'}, inplace=True)
        df = pd.merge(df, carga_total, on='Data', how='left')
        # Preencher eventuais NaN com a mediana
        df['Carga_Total_MWmed'] = df['Carga_Total_MWmed'].fillna(df['Carga_Total_MWmed'].median())
        print(f"    ✅ Carga incorporada ({df['Carga_Total_MWmed'].notna().sum()} meses)")
    
    # --- ENGENHARIA DE FEATURES: Lagged Features ---
    cols_volume = ['Vol_SE_CO', 'Vol_NE', 'Vol_Sul', 'Vol_Norte']
    for col in cols_volume:
        df[f'{col}_Lag1'] = df[col].shift(1)   # Mês anterior
        df[f'{col}_Lag2'] = df[col].shift(2)   # 2 meses atrás
        df[f'{col}_Delta'] = df[col].diff()     # Variação mensal (enchendo ou secando?)
    
    # Remover linhas sem histórico (primeiras 2)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # --- SEPARAR FEATURES (X) E TARGET (y) ---
    colunas_features = [c for c in df.columns if c not in ['Data', 'Target']]
    X = df[colunas_features]
    y = df['Target'].astype(int)
    
    print(f"\n  📐 Features criadas: {len(colunas_features)}")
    for f in colunas_features:
        print(f"      • {f}")
    
    # --- DIVISÃO TREINO/TESTE (70/30) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    
    # --- NORMALIZAÇÃO SEGURA (Z-Score) ---
    # fit SOMENTE no treino → transform em ambos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n  ✅ Divisão: {X_train.shape[0]} meses treino | {X_test.shape[0]} meses teste")
    print(f"  ✅ Normalização Z-Score aplicada SEM Data Leakage")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, colunas_features


# =============================================================================
# FASE 3: MODELAGEM — 3 ALGORITMOS (Exigência mínima da AV2)
# =============================================================================
def treinar_modelos(X_train, y_train):
    """
    Treina 3 modelos de ML exigidos pela AV2.
    
    CONCEITOS-CHAVE:
    
    1. class_weight='balanced': O sklearn calcula automaticamente pesos
       inversamente proporcionais à frequência de cada classe. Assim,
       errar uma Vermelha P2 (rara) custa MUITO mais que errar uma Verde.
    
    2. Regressão Logística Multinomial: Usa a função Softmax para calcular
       a probabilidade de cada bandeira. É o modelo mais interpretável.
    
    3. Random Forest: Conjunto de centenas de árvores de decisão independentes.
       Resistente a outliers e fornece Feature Importance.
    
    4. SVM com kernel RBF: Traça fronteiras de decisão curvilíneas.
       Ideal para dados que não são linearmente separáveis.
    """
    print("\n" + "=" * 70)
    print("  FASE 3: MODELAGEM — TREINANDO 3 ALGORITMOS")
    print("=" * 70)
    
    modelos = {}
    
    # Modelo 1: Regressão Logística (Baseline)
    print("\n  🔵 Treinando Modelo 1: Regressão Logística (Multinomial)...")
    lr = LogisticRegression(
        class_weight='balanced',  # Penaliza erros nas classes raras
        max_iter=2000,            # Garante convergência
        solver='lbfgs',           # Usa Softmax automaticamente para 4 classes
        random_state=SEED
    )
    lr.fit(X_train, y_train)
    modelos['Regressão Logística'] = lr
    print("    ✅ Hiperparâmetros: class_weight=balanced, max_iter=2000, solver=lbfgs")
    
    # Modelo 2: Random Forest
    print("\n  🌲 Treinando Modelo 2: Random Forest...")
    rf = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=300,      # 300 árvores de decisão
        max_depth=10,          # Limita profundidade para evitar overfitting
        min_samples_leaf=3,    # Cada folha precisa de pelo menos 3 amostras
        random_state=SEED
    )
    rf.fit(X_train, y_train)
    modelos['Random Forest'] = rf
    print("    ✅ Hiperparâmetros: n_estimators=300, max_depth=10, min_samples_leaf=3")
    
    # Modelo 3: SVM (Support Vector Machine) com kernel RBF
    print("\n  🔴 Treinando Modelo 3: SVM (kernel RBF)...")
    svm = SVC(
        class_weight='balanced',
        kernel='rbf',        # Kernel Radial — fronteiras curvilíneas
        C=1.0,               # Regularização (penalidade por erros)
        gamma='scale',       # Escala automática baseada nas features
        probability=True,    # Necessário para ROC e probabilidades
        random_state=SEED
    )
    svm.fit(X_train, y_train)
    modelos['SVM (RBF)'] = svm
    print("    ✅ Hiperparâmetros: kernel=rbf, C=1.0, gamma=scale")
    
    return modelos


# =============================================================================
# FASE 4: VALIDAÇÃO RIGOROSA (25% da nota — MAIOR PESO)
# =============================================================================
def validar_modelos(modelos, X_train, X_test, y_train, y_test, colunas_features):
    """
    Validação completa com:
    - Relatório de classificação (Precision, Recall, F1-Score)
    - Validação Cruzada Estratificada (5-Fold)
    - Matrizes de Confusão visuais
    - Curva ROC Multiclasse
    - Teste Estatístico de McNemar
    """
    print("\n" + "=" * 70)
    print("  FASE 4: VALIDAÇÃO RIGOROSA")
    print("=" * 70)
    
    resultados = {}
    previsoes = {}
    
    relatorio_path = os.path.join(PASTA_RESULTADOS, 'relatorio_metricas.txt')
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  RELATÓRIO DE MÉTRICAS — PROJETO ML: BANDEIRAS TARIFÁRIAS\n")
        f.write("=" * 70 + "\n\n")
        
        # --- AVALIAÇÃO DE CADA MODELO ---
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        
        for nome, modelo in modelos.items():
            print(f"\n  📊 Avaliando: {nome}")
            
            # Previsões no teste
            pred = modelo.predict(X_test)
            previsoes[nome] = pred
            
            # Métricas
            macro_f1 = f1_score(y_test, pred, average='macro', zero_division=0)
            weighted_f1 = f1_score(y_test, pred, average='weighted', zero_division=0)
            resultados[nome] = {'macro_f1': macro_f1, 'weighted_f1': weighted_f1}
            
            # Relatório completo
            report = classification_report(
                y_test, pred,
                target_names=NOMES_BANDEIRAS,
                labels=[0, 1, 2, 3],
                zero_division=0
            )
            
            # Validação Cruzada no TREINO
            cv_scores = cross_val_score(
                modelo, X_train, y_train, cv=cv_strategy, scoring='f1_macro'
            )
            resultados[nome]['cv_mean'] = cv_scores.mean()
            resultados[nome]['cv_std'] = cv_scores.std()
            
            # Imprimir e salvar
            texto = f"\n{'─' * 50}\n"
            texto += f"  {nome}\n"
            texto += f"{'─' * 50}\n"
            texto += f"  Macro F1-Score (Teste):    {macro_f1:.3f}\n"
            texto += f"  Weighted F1-Score (Teste): {weighted_f1:.3f}\n"
            texto += f"  CV 5-Fold Macro F1:        {cv_scores.mean():.3f} (±{cv_scores.std():.3f})\n\n"
            texto += report + "\n"
            
            print(texto)
            f.write(texto)
        
        # --- MATRIZES DE CONFUSÃO (3 modelos lado a lado) ---
        print("  📊 Gerando Matrizes de Confusão...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cmaps = ['Blues', 'Greens', 'Reds']
        for i, (nome, pred) in enumerate(previsoes.items()):
            ConfusionMatrixDisplay.from_predictions(
                y_test, pred, display_labels=NOMES_BANDEIRAS,
                cmap=cmaps[i], ax=axes[i], colorbar=False
            )
            f1 = resultados[nome]['macro_f1']
            axes[i].set_title(f'{nome}\nMacro F1 = {f1:.3f}', fontsize=11)
        plt.suptitle('Matrizes de Confusão — Comparação dos 3 Modelos', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(PASTA_GRAFICOS, '08_matrizes_confusao.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("    ✅ Salvo: 08_matrizes_confusao.png")
        
        # --- TESTE DE McNEMAR ---
        # Compara os dois melhores modelos para ver se a diferença é estatisticamente significativa
        nomes_modelos = list(previsoes.keys())
        melhor_idx = max(range(len(nomes_modelos)), key=lambda i: resultados[nomes_modelos[i]]['macro_f1'])
        segundo_idx = max(
            [i for i in range(len(nomes_modelos)) if i != melhor_idx],
            key=lambda i: resultados[nomes_modelos[i]]['macro_f1']
        )
        melhor_nome = nomes_modelos[melhor_idx]
        segundo_nome = nomes_modelos[segundo_idx]
        pred_a = previsoes[melhor_nome]
        pred_b = previsoes[segundo_nome]
        
        # Montar tabela de contingência
        ab, a_nb, na_b, na_nb = 0, 0, 0, 0
        for yt, pa, pb in zip(y_test, pred_a, pred_b):
            if pa == yt and pb == yt: ab += 1
            elif pa == yt and pb != yt: a_nb += 1
            elif pa != yt and pb == yt: na_b += 1
            else: na_nb += 1
        
        tabela = [[ab, a_nb], [na_b, na_nb]]
        resultado_mcnemar = mcnemar(tabela, exact=True)
        
        texto_mcnemar = f"\n{'=' * 50}\n"
        texto_mcnemar += f"  TESTE ESTATÍSTICO DE McNEMAR\n"
        texto_mcnemar += f"{'=' * 50}\n"
        texto_mcnemar += f"  Comparando: {melhor_nome} vs {segundo_nome}\n"
        texto_mcnemar += f"  Tabela de Contingência: {tabela}\n"
        texto_mcnemar += f"  P-Value: {resultado_mcnemar.pvalue:.4f}\n"
        if resultado_mcnemar.pvalue < 0.05:
            texto_mcnemar += f"  ➡️  Conclusão: Diferença ESTATISTICAMENTE SIGNIFICATIVA (p < 0.05)\n"
            texto_mcnemar += f"  ➡️  O modelo '{melhor_nome}' é comprovadamente superior.\n"
        else:
            texto_mcnemar += f"  ➡️  Conclusão: Diferença NÃO significativa (p ≥ 0.05)\n"
            texto_mcnemar += f"  ➡️  Pela regra de parcimônia, escolhe-se o modelo mais simples.\n"
        
        print(texto_mcnemar)
        f.write(texto_mcnemar)
    
    print(f"\n  ✅ Relatório salvo em: {relatorio_path}")
    return resultados, previsoes


# =============================================================================
# FASE 5: INTERPRETAÇÃO E INSIGHTS (20% da nota)
# =============================================================================
def gerar_interpretacao(modelos, colunas_features, resultados):
    """
    Gera os gráficos de Feature Importance e documenta insights de negócio.
    """
    print("\n" + "=" * 70)
    print("  FASE 5: INTERPRETAÇÃO E INSIGHTS DE NEGÓCIO")
    print("=" * 70)
    
    rf = modelos['Random Forest']
    
    # --- GRÁFICO: Feature Importance (Random Forest) ---
    importancias = pd.Series(rf.feature_importances_, index=colunas_features).sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cores = ['#e74c3c' if 'SE_CO' in f else '#3498db' if 'NE' in f 
             else '#27ae60' if 'Sul' in f else '#9b59b6' if 'Norte' in f 
             else '#f39c12' for f in importancias.index]
    importancias.plot(kind='barh', color=cores, ax=ax, edgecolor='black', linewidth=0.3)
    ax.set_title('Importância das Variáveis na Decisão da Bandeira\n(Random Forest — Feature Importance)', fontsize=13)
    ax.set_xlabel('Peso da Variável')
    
    # Legenda de cores
    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(color='#e74c3c', label='Sudeste/CO'),
        mpatches.Patch(color='#3498db', label='Nordeste'),
        mpatches.Patch(color='#27ae60', label='Sul'),
        mpatches.Patch(color='#9b59b6', label='Norte'),
        mpatches.Patch(color='#f39c12', label='Carga/Outro'),
    ]
    ax.legend(handles=legend, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, '09_feature_importance.png'), dpi=150)
    plt.close()
    print("  📊 Gráfico Feature Importance salvo")
    
    # --- INSIGHTS DE NEGÓCIO ---
    # Tabela de custo por bandeira (valores ANEEL 2024/2025)
    custo_por_100kwh = {0: 0.00, 1: 1.885, 2: 4.463, 3: 7.877}
    consumo_medio_kwh = 150  # Consumo médio residencial brasileiro
    
    print("\n  💡 INSIGHTS DE NEGÓCIO PARA O RELATÓRIO:")
    print("  " + "─" * 55)
    
    top3 = importancias.tail(3).index.tolist()[::-1]
    print(f"\n  1️⃣  As 3 variáveis mais importantes são:")
    for i, feat in enumerate(top3):
        print(f"      #{i+1}: {feat} (peso: {importancias[feat]:.3f})")
    
    print(f"\n  2️⃣  Impacto Financeiro por Bandeira (consumo médio {consumo_medio_kwh} kWh/mês):")
    for b, nome in enumerate(NOMES_BANDEIRAS):
        custo = custo_por_100kwh[b] * (consumo_medio_kwh / 100)
        print(f"      {nome}: R$ {custo:.2f} de acréscimo na conta")
    
    print(f"\n  3️⃣  Prever corretamente uma Vermelha P2 evita que consumidores")
    print(f"      e empresas sejam surpreendidos por um acréscimo de até")
    print(f"      R$ {custo_por_100kwh[3] * (consumo_medio_kwh/100):.2f}/mês na conta de luz.")
    
    print(f"\n  4️⃣  LIMITAÇÕES DO MODELO:")
    print(f"      • Dados limitados: apenas ~130 meses de histórico")
    print(f"      • Classes raras: Vermelha P2/Escassez são eventos extremos")
    print(f"      • Não captura fatores políticos/regulatórios da ANEEL")
    print(f"      • Mudanças climáticas podem alterar padrões históricos")
    
    melhor = max(resultados, key=lambda k: resultados[k]['macro_f1'])
    print(f"\n  5️⃣  MODELO RECOMENDADO: {melhor}")
    print(f"      Macro F1 = {resultados[melhor]['macro_f1']:.3f}")
    print(f"      CV 5-Fold = {resultados[melhor]['cv_mean']:.3f} (±{resultados[melhor]['cv_std']:.3f})")
    
    # Salvar insights em arquivo
    with open(os.path.join(PASTA_RESULTADOS, 'relatorio_metricas.txt'), 'a', encoding='utf-8') as f:
        f.write("\n\n" + "=" * 50 + "\n")
        f.write("  INSIGHTS DE NEGÓCIO\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nTop 3 Features: {', '.join(top3)}\n")
        f.write(f"Modelo Recomendado: {melhor} (Macro F1 = {resultados[melhor]['macro_f1']:.3f})\n")
        f.write(f"Validação Cruzada: {resultados[melhor]['cv_mean']:.3f} ± {resultados[melhor]['cv_std']:.3f}\n")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    print("\n" + "🔬" * 35)
    print("  PROJETO ML — PREVISÃO DE BANDEIRAS TARIFÁRIAS")
    print("  Etapas 2 a 5: Preparação → Modelagem → Validação → Insights")
    print("🔬" * 35)
    
    # Fase 2
    X_train, X_test, y_train, y_test, colunas_features = preparar_dados()
    
    # Fase 3
    modelos = treinar_modelos(X_train, y_train)
    
    # Fase 4
    resultados, previsoes = validar_modelos(
        modelos, X_train, X_test, y_train, y_test, colunas_features
    )
    
    # Fase 5
    gerar_interpretacao(modelos, colunas_features, resultados)
    
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETO FINALIZADO!")
    print(f"  📁 Gráficos em: {PASTA_GRAFICOS}")
    print(f"  📄 Relatório em: {PASTA_RESULTADOS}")
    print("=" * 70)
