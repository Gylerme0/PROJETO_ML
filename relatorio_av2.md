# Relatório Técnico — Previsão de Bandeiras Tarifárias de Energia Elétrica

## 1. Introdução

Na gestão de Supply Chain e no planejamento financeiro de indústrias e distribuidoras de energia, a antecipação de custos é fundamental. O Sistema Interligado Nacional (SIN) brasileiro possui forte dependência de fontes hidrológicas. Quando há escassez de chuvas e queda no volume dos reservatórios, o Operador Nacional do Sistema (ONS) despacha usinas termelétricas, que possuem um custo variável unitário (CVU) substancialmente superior. Para compensar essa elevação de custos e sinalizar o consumidor, a Agência Nacional de Energia Elétrica (ANEEL) aciona o mecanismo das Bandeiras Tarifárias.

O objetivo deste trabalho é desenvolver um modelo de Machine Learning focado em **Classificação Multiclasse** para prever com precisão o acionamento das bandeiras tarifárias (Verde, Amarela, Vermelha Patamar 1 e Vermelha Patamar 2) com base em dados hidrológicos e de consumo, conferindo especial destaque ao comportamento do reservatório de Sobradinho e demais subsistemas do SIN.

## 2. Descrição dos Dados e Engenharia de Features

Os dados foram obtidos por meio de extração direta das bases de Dados Abertos do governo federal (ANEEL e ONS) e estruturados em um banco de dados local (SQLite).

### Bases Utilizadas:
- **Variável-Alvo (Target):** Histórico estruturado de acionamentos mensais da Bandeira Tarifária (ANEEL) de 2015 a 2026.
- **Variáveis Preditoras:** Volume útil diário consolidado (%) dos 4 subsistemas nacionais (ONS) agregados mensalmente, e Carga de Energia Mensal em MWmed (ONS).

### O Desafio do Desbalanceamento
A etapa de Análise Exploratória de Dados (EDA) evidenciou um acentuado desbalanceamento: cerca de 49.3% dos meses da amostra operam na normalidade hídrica (Bandeira Verde), enquanto os eventos críticos (Vermelha P2 e Escassez) correspondem a apenas 16.9%. Isso exigiu a adaptação das métricas de avaliação, priorizando o F1-Score em detrimento da acurácia global.

### Engenharia de Features (Lagged Features)
Para que o algoritmo fosse capaz de aprender a inércia hidrológica (o tempo que as bacias demoram para encher ou secar), foram criadas *Lag Features*: variáveis representando o volume dos reservatórios no mês anterior (Lag1) e há dois meses (Lag2), além da variação (Delta). Os dados com valores extremos causados por falhas nos sensores do ONS (ex: valores fora da margem 0-110%) foram limpos e imputados para preservar a qualidade da modelagem.

## 3. Metodologia de Modelagem e Validação

Para endereçar o problema de classificação multiclasse sob forte desbalanceamento, aplicou-se o método de estratificação na separação de Treino (70%) e Teste (30%). Para evitar o fenômeno de *Data Leakage* (vazamento de dados), a padronização das variáveis via Z-Score (StandardScaler) ocorreu apenas sobre as estatísticas do grupo de treino.

Foram treinados três algoritmos com penalização proporcional para classes raras (`class_weight='balanced'`):
1. **Regressão Logística Multinomial:** Baseline de máxima interpretabilidade.
2. **Random Forest Classifier:** Ensemble robusto não linear, resistente a anomalias residuais.
3. **Máquina de Vetores de Suporte (SVM) com Kernel RBF:** Mapeamento em altas dimensões para traçar fronteiras complexas.

A etapa de Validação Rigorosa baseou-se na técnica de **Validação Cruzada Estratificada em 5 partições (5-Fold Stratified CV)** sobre o conjunto de treino. Como métrica principal, adotou-se o Macro F1-Score, por sua resiliência a classes minoritárias. Para inferir a significância estatística de eventual superioridade entre os modelos, aplicou-se o **Teste de McNemar**.

## 4. Resultados e Análise

Os resultados aferidos nos dados de teste atestam o desempenho das arquiteturas:

- **Regressão Logística:** Macro F1 = 0.471 (CV 5-Fold = 0.308 ±0.046)
- **Random Forest:** Macro F1 = 0.446 (CV 5-Fold = 0.366 ±0.102)
- **SVM (RBF):** Macro F1 = 0.432 (CV 5-Fold = 0.353 ±0.128)

**Teste Estatístico e Escolha do Modelo:**
Ao aplicar o teste de McNemar na matriz de contingência comparando Regressão Logística e Random Forest, obteve-se um p-value igual a 1.000. Isso indica que não há evidências de diferença estatisticamente significativa no desempenho entre eles. Pelo princípio da parcimônia (Navalha de Ockham), recomendou-se a **Regressão Logística**, dada a sua maior interpretabilidade e baixo custo computacional.

## 5. Interpretação, Insights de Negócio e Limitações

A modelagem evidenciou premissas vitais para a antecipação de custos de energia:

1. **Os Vetores da Crise:** O gráfico de *Feature Importance* (obtido do Random Forest) prova que o Volume do subsistema Sul (com Lags de 1 e 2 meses) e do Norte exerceram impacto formidável no acionamento recente de bandeiras. Sobradinho (Nordeste) atua como um regulador sensível frente às matrizes da região Sudeste.
2. **Impacto Financeiro:** Um modelo capaz de diagnosticar e prever corretamente a entrada em Bandeira Vermelha P2 permite que empresas e indústrias realizem hedge energético antes da escalada dos preços no mercado livre, contornando a incidência de até R$ 11.82 a mais na conta de luz de um consumidor médio.

**Limitações Diagnosticadas:**
- A base temporal disponível do mecanismo oficial de bandeiras possui em torno de 130 amostras, o que torna eventos extremos (Vermelha P2) estatisticamente muito raros, dificultando o aprendizado pleno por algoritmos complexos como o Random Forest.
- Fatores exógenos não hidrológicos, como regulações intempestivas ou interferências políticas do Governo na ANEEL, não são apreendidos pela matemática hidrológica do modelo.

## 6. Conclusão

A pesquisa atesta que algoritmos lineares simples, quando associados a uma robusta Engenharia de Dados na extração de janelas regressivas (Lagged Features), competem pare a pare com arquiteturas complexas na modelagem do setor elétrico. A Regressão Logística atingiu o melhor balanceamento analítico e computacional, provendo uma ferramenta valiosa de planejamento estratégico preditivo contra flutuações tarifárias sazonais.
