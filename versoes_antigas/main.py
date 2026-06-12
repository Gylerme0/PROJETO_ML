# =============================================================================
# MAIN.PY — PIPELINE COMPLETO DE MACHINE LEARNING
# =============================================================================
# Título: "Previsão de Bandeiras Tarifárias e Avaliação de Impacto Financeiro:
#          Um Modelo de Classificação Baseado em Dados Hidrometeorológicos"
#
# Disciplina: Machine Learning — AV2
# Professor: Mateus Silva
#
# DESCRIÇÃO:
#   Este script executa todas as etapas do projeto de ML de ponta a ponta:
#     1. Análise Exploratória de Dados (EDA) — 7 gráficos
#     2. Preparação dos Dados (Features, Split, Normalização)
#     3. Modelagem (3 algoritmos: Regressão Logística, Random Forest, SVM)
#     4. Validação Rigorosa (CV, McNemar, Matrizes de Confusão)
#     5. Interpretação e Insights de Negócio
#
# PRÉ-REQUISITOS:
#   - Executar scraper.py primeiro para gerar o banco base_energia.db
#   - pip install pandas numpy scikit-learn matplotlib seaborn statsmodels pyarrow
#
# COMO EXECUTAR:
#   python main.py
# =============================================================================

from etapa1_eda import carregar_dados, gerar_graficos_eda
from etapa2_modelagem import preparar_dados, treinar_modelos, validar_modelos, gerar_interpretacao

def main():
    print("\n" + "=" * 70)
    print("  🎓 PROJETO ML — PREVISÃO DE BANDEIRAS TARIFÁRIAS")
    print("  📋 Pipeline Completo: EDA → Preparação → Modelagem → Validação")
    print("=" * 70)
    
    # =========================================================================
    # ETAPA 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA) — Fase 1 da AV2 (15%)
    # =========================================================================
    print("\n\n" + "🔍" * 35)
    print("  ETAPA 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("🔍" * 35)
    df = carregar_dados()
    gerar_graficos_eda(df)
    
    # =========================================================================
    # ETAPA 2: PREPARAÇÃO + MODELAGEM + VALIDAÇÃO + INTERPRETAÇÃO
    # Fases 2 (Prep), 3 (20%), 4 (25%) e 5 (20%) da AV2
    # =========================================================================
    X_train, X_test, y_train, y_test, features = preparar_dados()
    modelos = treinar_modelos(X_train, y_train)
    resultados, previsoes = validar_modelos(
        modelos, X_train, X_test, y_train, y_test, features
    )
    gerar_interpretacao(modelos, features, resultados)
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  ✅ PROJETO ML FINALIZADO COM SUCESSO!")
    print("=" * 70)
    print("\n  📁 Arquivos gerados:")
    print("      graficos/01_distribuicao_bandeiras.png  — Desbalanceamento")
    print("      graficos/02_correlacao_regioes.png      — Correlação")
    print("      graficos/03_boxplot_ne_vs_bandeira.png  — Sobradinho vs Bandeira")
    print("      graficos/04_evolucao_historica.png      — Série Temporal")
    print("      graficos/05_outliers_volume.png         — Outliers")
    print("      graficos/06_dispersao_fronteira.png     — Fronteira de Decisão")
    print("      graficos/07_timeline_bandeiras.png      — Linha do Tempo")
    print("      graficos/08_matrizes_confusao.png       — Matrizes de Confusão")
    print("      graficos/09_feature_importance.png      — Importância das Variáveis")
    print("      resultados/relatorio_metricas.txt       — Relatório Completo")
    print("\n  📝 Para o relatório PDF e slides, consulte o relatorio_metricas.txt")
    print("=" * 70)


if __name__ == '__main__':
    main()