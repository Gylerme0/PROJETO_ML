# Decisões Técnicas e Metodológicas

Este documento (exigência RQ-DO-02) registra as escolhas arquiteturais do projeto.

### 1. Pré-processamento e Imputação
- **Escolha:** Interpolação Linear e Média Móvel (Rolling Windows).
- **Justificativa:** Os dados do INMET e ONS possuem falhas (dias sem coleta). A interpolação linear mantém a coerência temporal da série. As médias móveis (30, 90, 365 dias) foram criadas para representar a "inércia hidrológica", pois a tarifa não responde à chuva de um único dia, mas ao acúmulo histórico.

### 2. Validação Cruzada (GroupKFold vs StratifiedKFold)
- **Trade-off:** O `StratifiedKFold` resultava em vazamento de dados (Data Leakage) atingindo 99% de acurácia, pois misturava dias do mesmo mês no treino e teste.
- **Decisão:** Optou-se pelo `GroupKFold` usando a variável `DataRef` (Mês/Ano) como grupo. Isso garante que o modelo treine em blocos temporais do passado para prever meses inteiros do futuro, refletindo um cenário real de predição.

### 3. Métricas de Avaliação
- **Escolha:** Acurácia Global e F1-Score (Macro).
- **Justificativa:** O problema é nativamente desbalanceado (muitos dias verdes, poucos de crise). O F1-Score macro garante que o modelo seja penalizado se ignorar a classe minoritária (que é a mais importante para o negócio).

### 4. Pivotagem de Multiclasse para Binário (Isenção vs Sobretaxa)
- **Trade-off (Desempenho vs Granularidade):** Modelos multiclasse (Verde vs Amarela vs P1 vs P2) esbarravam em 65% de acurácia devido ao ruído discricionário entre Verde e Amarela.
- **Justificativa:** Para o setor de energia, o impacto financeiro (dor) ocorre ao sair da Isenção (Verde) para a Sobretaxa (qualquer outra). Agrupar o alvo em Binário eliminou o ruído e elevou o desempenho preditivo para a casa dos 82%.

### 5. Algoritmo Selecionado
- **Escolha:** XGBoost (eXtreme Gradient Boosting).
- **Justificativa:** Superou a Regressão Logística e o Random Forest em testes empíricos (comprovado pelo Teste de McNemar). Árvores de Gradiente lidam perfeitamente com interações não lineares entre Anomalias de Chuva e Volume.
