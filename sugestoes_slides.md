# Correções e Sugestões de Melhoria para os Slides

Textos prontos para copiar e colar. Cada seção indica o slide correspondente.

---

## 🔴 SLIDE 3 — CORREÇÃO OBRIGATÓRIA

### Texto ERRADO atual:
> Isenção Tarifária: Bandeiras Verde e Amarela — sem sobretaxa ao consumidor final
> Sobretaxa: Bandeiras Vermelha 1 e 2 — acréscimo direto na fatura de energia

### Texto CORRETO (substituir):

Isenção Tarifária (Classe 0)
Somente Bandeira Verde — condições hídricas favoráveis, sem acréscimo na fatura

Sobretaxa (Classe 1)
Bandeiras Amarela, Vermelha P1 e Vermelha P2 — acréscimo direto na fatura de energia, com severidade escalonada

---

## SLIDE 4 — Tratamento de Dados Faltantes

### Adicionar ao slide (novo tópico ou nota):

Tratamento de Dados
Os dados brutos são armazenados em banco SQLite e arquivos Excel (ENA). Falhas de coleta nos sensores dos reservatórios foram tratadas via interpolação linear bidirecional, garantindo continuidade temporal sem introduzir viés estatístico.

---

## SLIDE 5 — Exemplo Concreto de Feature

### Adicionar ao slide (caixa de destaque ou nota):

Exemplo de Feature Construída:

Vol_SE_CO_anomalia_1ano = (Volume Atual − Média Móvel 365 dias) / Média Móvel 365 dias

Interpretação: Valor negativo indica que o reservatório do Sudeste/Centro-Oeste está abaixo da sua média histórica anual. Quanto mais negativo, maior o estresse hídrico e maior a probabilidade de sobretaxa.

Outras features derivadas:
• Médias móveis de 30, 90 e 365 dias sobre os volumes dos 4 subsistemas (SE/CO, Sul, Norte, Nordeste)
• Sazonalidade cíclica: sin(2π × dia/365) e cos(2π × dia/365)
• Mês hidrológico codificado (outubro = mês 1, setembro = mês 12)
• Anomalia ENA: média de 60 dias da ENA (% MLT) − 100 → indica quanto abaixo do ideal histórico

---

## SLIDE 6 — Hiperparâmetros dos Modelos

### Adicionar abaixo de cada modelo:

Regressão Logística (Baseline)
• max_iter = 1.000
• class_weight = 'balanced' (compensação automática do desbalanceamento)
• Solver padrão (lbfgs)

Random Forest (Ensemble)
• n_estimators = 100 árvores
• max_depth = 5
• class_weight = 'balanced'

XGBoost (Modelo Campeão)
• n_estimators = 300 árvores de gradiente
• max_depth = 5
• learning_rate = 0,05
• subsample = 0,8 (80% das amostras por árvore)
• colsample_bytree = 0,8 (80% das features por árvore)

---

## SLIDE 7 — Normalização (StandardScaler)

### Adicionar ao slide (novo tópico):

Normalização Z-Score
Antes do treinamento, todas as features foram normalizadas via StandardScaler (média = 0, desvio-padrão = 1). O scaler foi ajustado exclusivamente nos dados de treino, garantindo que nenhuma estatística do conjunto de teste contaminasse o processo — outra camada de proteção contra Data Leakage.

---

## SLIDE 8 — Tabela Completa de Métricas

### Adicionar ao slide (tabela):

Modelo XGBoost — Relatório de Classificação Final

                              Precision   Recall   F1-Score   Support
Isenção (Verde)                 0,810     0,836     0,823     2.033
Sobretaxa (Amarela/Vermelha)    0,837     0,810     0,823     2.105

Acurácia Global                                     0,823     4.138
Macro Avg                       0,823     0,823     0,823     4.138

Validação: GroupKFold com 5 splits agrupados por mês
Teste Estatístico: McNemar (XGBoost vs Random Forest) — p < 0,05

---

## SLIDE 9 — Feature Importance

### Adicionar ao slide (lista das principais variáveis):

Top Variáveis Preditivas (Feature Importance do XGBoost):

1. Anomalia do Volume SE/CO (desvio em relação à média anual)
2. ENA SE/CO (% da Média de Longo Termo)
3. Volume SE/CO — Média Móvel de 90 dias
4. Volume Sul — Média Móvel de 365 dias
5. Anomalia ENA (desvio de 60 dias em relação a 100% MLT)

Conclusão: O subsistema Sudeste/Centro-Oeste, que abriga os maiores reservatórios regulatórios do país (incluindo Itaipu, Furnas e Três Marias), domina a predição. O estresse hídrico nessa região é o principal gatilho das bandeiras tarifárias.

Suavização Mensal (Regra de Negócio do ONS):
A bandeira é definida uma vez por mês. O modelo gera predições diárias e aplica a Moda intra-mensal — a classe mais votada entre os 30 dias define a predição final do mês, eliminando flutuações isoladas.
