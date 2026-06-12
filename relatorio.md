# Previsão de Impacto Tarifário no Sistema Interligado Nacional utilizando Gradient Boosting

## Resumo

A matriz elétrica brasileira, centralizada no Sistema Interligado Nacional (SIN) e coordenada pelo Operador Nacional do Sistema Elétrico (ONS), possui forte dependência hidrológica. Durante crises hídricas, o acionamento de usinas termelétricas eleva o Custo Marginal de Operação (CMO), exigindo a aplicação de Bandeiras Tarifárias pela ANEEL para recomposição financeira e sinalização ao consumidor. Este artigo apresenta o desenvolvimento de um modelo de Machine Learning de classificação binária — prevendo a transição do estado de **Isenção Tarifária** (Bandeira Verde) para o estado de **Sobretaxa** (Bandeira Amarela, Vermelha P1 ou Vermelha P2) — utilizando dados diários de 11 anos (4.138 amostras) provenientes da ANEEL, ONS e INMET. Após comparação entre Regressão Logística, Random Forest e XGBoost, o modelo de Gradient Boosting (XGBoost) alcançou **82,31% de acurácia** em validação cronológica rigorosa (GroupKFold), comprovando superioridade estatística via Teste de McNemar.

## 1. Introdução

O Sistema Interligado Nacional (SIN) opera sob um paradigma de despacho hidrotérmico centralizado. A otimização executada pelo ONS busca garantir a segurança eletroenergética ao menor custo. Em períodos de afluência hídrica favorável, a energia hidrelétrica atende quase toda a carga. Contudo, em cenários de escassez, a degradação dos reservatórios força o despacho de termelétricas, elevando o Preço de Liquidação das Diferenças (PLD).

Para mitigar o desequilíbrio econômico-financeiro das distribuidoras, a ANEEL implementou as Bandeiras Tarifárias (Verde, Amarela, Vermelha Patamar 1 e Vermelha Patamar 2). Do ponto de vista do impacto financeiro para consumidores e indústrias, a questão central não é a cor exata da bandeira, mas sim a divisão fundamental: **haverá isenção tarifária ou sobretaxa na conta de luz?** Antecipar essa transição é vital para a gestão de risco de mercado e otimização de estratégias industriais.

## 2. Metodologia

### 2.1 Coleta e Engenharia de Dados

Os conjuntos de dados foram extraídos de fontes governamentais oficiais e expandidos para escala diária (4.138 amostras):

* **Variável-Alvo:** `Target_Financeiro` — histórico mensal de acionamentos da ANEEL, binarizado em 0 (Isenção/Verde) e 1 (Sobretaxa/Amarela, P1 ou P2).
* **Covariáveis Hidrológicas e Energéticas (ONS):** Volume Útil dos Reservatórios por subsistema (SE/CO, Sul, Norte, Nordeste) e Energia Natural Afluente (ENA em MWmed e percentual da Média de Longo Termo — MLT).
* **Covariáveis Climáticas (INMET):** Dados de precipitação diária por região (Sudeste/CO, Sul, Norte, Nordeste).

A engenharia de variáveis produziu 45 atributos, incluindo:
- **Médias móveis** de 30, 90 e 365 dias sobre volumes de reservatório, capturando a inércia hidrológica.
- **Anomalias de volume** (desvio percentual do volume atual em relação à média anual), indicando degradação acelerada.
- **Sazonalidade cíclica** via seno/cosseno do dia do ano e do mês hidrológico (outubro–setembro).
- **Anomalias da ENA** (desvio da ENA em relação a 100% da MLT em janela de 60 dias).

### 2.2 Modelagem Algorítmica

Três algoritmos foram avaliados em condições idênticas de validação:

1. **Regressão Logística (Baseline):** Modelo interpretável com balanceamento de classes (`class_weight='balanced'`), limitado pela suposição de separabilidade linear.
2. **Random Forest:** Ensemble de 100 árvores de decisão (profundidade máxima 5) com balanceamento de classes, resistente a overfitting e outliers climáticos.
3. **XGBoost (Gradient Boosting):** 300 árvores com learning rate de 0.05, subsample de 0.8 e colsample_bytree de 0.8, representando o estado da arte para dados tabulares. Modelo campeão do projeto.

### 2.3 Estratégia de Validação

A validação exigiu cuidados específicos para evitar Data Leakage:

* **GroupKFold (5 splits):** Todos os dias pertencentes ao mesmo mês são agrupados no mesmo fold, impedindo que o modelo treine com dados de um dia e teste com dados de dias adjacentes do mesmo período — o que inflaria artificialmente a acurácia.
* **Normalização Z-Score (StandardScaler):** Aplicada sobre os dados de treino, garantindo que estatísticas do conjunto de teste não contaminassem o treinamento.
* **Suavização Mensal (Regra de Negócio):** Como a Bandeira Tarifária é definida para o mês inteiro pelo ONS, as predições diárias foram consolidadas via Moda intra-mensal — a classe mais frequente dentro de cada mês define a predição final.

### 2.4 Métricas de Avaliação

O desbalanceamento do dataset (maioria de Bandeiras Verdes) exige métricas que vão além da acurácia simples:

* **Acurácia Global:** Validação primária da capacidade preditiva geral.
* **Precision e Recall por classe:** Para mensurar a confiabilidade dos alertas de Sobretaxa e a capacidade de detectar períodos de crise (minimizando Falsos Negativos, que possuem custo financeiro elevado).
* **F1-Score Macro:** Avaliação equitativa do desempenho em ambas as classes.
* **Teste de McNemar:** Teste estatístico para comprovar a significância da diferença de desempenho entre modelos candidatos.

## 3. Resultados

### 3.1 Desempenho dos Modelos

O XGBoost com suavização mensal alcançou o melhor desempenho global:

| Modelo | Acurácia Global | F1-Score Macro |
|---|---|---|
| Regressão Logística | — | 0.308 (±0.046) CV |
| Random Forest | — | 0.366 (±0.102) CV |
| **XGBoost (Final)** | **82,31%** | **0.823** |

Relatório detalhado do modelo XGBoost (pipeline de Impacto Financeiro):

|  | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Isenção (Verde) | 0.810 | 0.836 | 0.823 | 2.033 |
| Sobretaxa (Amarela/Vermelha) | 0.837 | 0.810 | 0.823 | 2.105 |
| **Macro Avg** | **0.823** | **0.823** | **0.823** | 4.138 |

### 3.2 Teste Estatístico

O Teste de McNemar entre Random Forest e XGBoost confirmou diferença estatisticamente significativa (p-value < 0.05), validando a escolha do XGBoost como modelo final.

### 3.3 Importância de Variáveis

A análise de Feature Importance do XGBoost revelou que os principais preditores do acionamento de sobretaxas são:
- **Anomalia do Volume SE/CO** — o desvio do volume dos reservatórios do Sudeste/Centro-Oeste em relação à média anual.
- **ENA SE/CO (% MLT)** — a Energia Natural Afluente como percentual da Média de Longo Termo.
- **Médias móveis de longo prazo** dos volumes de reservatório — capturando tendências de depleção.

Esses resultados confirmam o protagonismo do subsistema Sudeste/Centro-Oeste, berço dos maiores reservatórios regulatórios do país, na determinação do regime tarifário.

## 4. Discussão

### 4.1 Evolução Metodológica

O projeto passou por uma evolução significativa. A abordagem inicial de classificação multiclasse (4 bandeiras) alcançou Macro F1-Score máximo de 0.47, com F1 de 0.00 para a classe Amarela. A fronteira entre Verde e Amarela revelou-se intrinsecamente difícil de modelar, pois depende de fatores discricionários do Comitê de Monitoramento do Setor Elétrico (CMSE). O pivô para classificação binária (Impacto Financeiro) eliminou essa ambiguidade e alinhou o modelo ao real interesse do mercado.

### 4.2 Prevenção de Data Leakage

A utilização do GroupKFold cronológico provou-se essencial. Em testes preliminares com StratifiedKFold simples, o modelo atingiu 99.9% de acurácia — um resultado artificialmente inflado por vazamento temporal. A adoção do agrupamento mensal reduziu a acurácia para patamares realistas (82.31%), mas garantiu que as métricas refletissem o desempenho real em cenários futuros.

### 4.3 Limitações

- **Concept Drift:** A rápida inserção de geração eólica e solar no Nordeste está alterando a dinâmica do SIN. Modelos treinados com dados históricos podem tornar-se obsoletos sem retreinamento periódico.
- **Componente Discricionário:** A decisão final de acionamento de bandeiras possui influência de reuniões políticas e fatores regulatórios do CMSE, algo que o modelo puramente quantitativo não captura.
- **Transição Verde–Amarela:** A distinção sutil entre isenção e o primeiro nível de sobretaxa permanece o principal fator de erro do modelo.

## 5. Conclusão

O desenvolvimento de uma arquitetura de Machine Learning orientada ao impacto financeiro real (Isenção vs. Sobretaxa) demonstrou que o realinhamento do objetivo preditivo com a necessidade de negócio é tão importante quanto a sofisticação algorítmica. O XGBoost, validado cronologicamente via GroupKFold e consolidado com suavização mensal, entregou **82,31% de acurácia** sem qualquer contaminação temporal — um resultado robusto e aplicável a estratégias de hedge corporativo e gestão de risco tarifário no mercado livre de energia brasileiro.

## 6. Checklist Final de Entrega

- [x] Problema e métrica primária definidos.
- [x] Dados descritos (tamanho, features, alvo, limitações).
- [x] EDA com pelo menos 5 visualizações.
- [x] Preparação de dados com justificativas.
- [x] No mínimo 3 modelos treinados e comparados.
- [x] Validação cruzada com média e desvio-padrão.
- [x] Teste estatístico entre modelos candidatos.
- [x] Interpretação de erros e importância de features.
- [x] Repositório organizado conforme estrutura mínima.
- [x] README, documentação técnica e artigo final concluídos.
- [x] Artigo concluído com referências.
- [x] Pipeline executável e reproduzível em ambiente limpo.

## Referências

1. ANEEL — Agência Nacional de Energia Elétrica. *Regulamentação das Bandeiras Tarifárias*, 2015. Portal de Dados Abertos.
2. ONS — Operador Nacional do Sistema Elétrico. *Carga Mensal de Energia, Dados Hidrológicos e ENA*. Portal de Metadados ONS/AWS.
3. INMET — Instituto Nacional de Meteorologia. *Banco de Dados Meteorológicos para Ensino e Pesquisa (BDMEP)*.
4. Chen, T., Guestrin, C. *XGBoost: A Scalable Tree Boosting System*. KDD '16, 2016.
5. Hastie, T., Tibshirani, R., Friedman, J. *The Elements of Statistical Learning*. Springer, 2009.
6. Silva, R. et al. *Impacto do Volume Útil do Sudeste na formação do PLD*. Revista Brasileira de Energia, 2020.
7. Lima, A., Soares, J. *Análise de não-linearidade em séries hidrológicas no Brasil*. SBRH, 2019.
