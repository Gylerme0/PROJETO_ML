"Atue como um Especialista em Ciência de Dados e Energia. Crie uma apresentação de slides (Pitch Deck) acadêmica e profissional sobre a 'Previsão de Impacto Tarifário no Sistema Interligado Nacional utilizando Gradient Boosting'. O tom deve ser técnico, porém acessível.

O projeto utiliza um modelo de Machine Learning binário (Classificação Binária: Isenção Tarifária vs. Sobretaxa) treinado com XGBoost, que alcançou 82,31% de acurácia em validação cronológica rigorosa (GroupKFold). Os dados cobrem 4.138 dias (11 anos) de histórico com 45 features de engenharia hidrológica.

A estrutura deve conter os seguintes slides:

Capa: Título 'Previsão de Impacto Tarifário no Sistema Interligado Nacional utilizando Gradient Boosting' e espaço para o nome dos autores.

O Problema: O SIN, a dependência hidrológica, as termelétricas e o impacto financeiro (CMO/PLD). O que são as Bandeiras Tarifárias da ANEEL e por que prevê-las é crucial para o mercado de energia.

A Decisão de Negócio: Explicar por que o projeto pivotou de classificação multiclasse (4 bandeiras) para classificação binária (Isenção vs. Sobretaxa). A fronteira entre Verde e Amarela é discricionária (influência política do ONS/CMSE), mas a fronteira entre 'ter ou não ter sobretaxa' é o que realmente importa financeiramente.

O Desafio de Dados: Fontes governamentais (ANEEL, ONS, INMET) e a integração de variáveis climáticas (precipitação regional), hidrológicas (Volume Útil dos reservatórios por subsistema) e energéticas (ENA e % da MLT). Expansão para escala diária (4.138 amostras).

Engenharia de Variáveis: 45 atributos construídos — médias móveis de 30/90/365 dias, anomalias do volume em relação à média anual, sazonalidade cíclica via seno/cosseno do ano hidrológico (outubro–setembro), anomalias da ENA vs. MLT.

Algoritmos Avaliados: Comparação de 3 modelos — Regressão Logística (Baseline interpretável), Random Forest (Ensemble resistente a outliers) e XGBoost (Gradient Boosting, estado da arte para dados tabulares, modelo campeão do projeto).

O Problema do Data Leakage: Explicar que ao usar dados diários, uma validação cruzada ingênua (StratifiedKFold) misturava dias do mesmo mês entre treino e teste, inflando artificialmente a acurácia para 99.9%. A solução foi o GroupKFold cronológico, que agrupa todos os dias de um mesmo mês no mesmo fold.

Resultados Concretos: Acurácia Global de 82,31%, F1-Score Macro de 0.823, Precision de 84% para Sobretaxa, Recall de 81% para Sobretaxa. Teste de McNemar confirmou superioridade estatística do XGBoost vs Random Forest.

Feature Importance e Interpretabilidade: As variáveis mais preditivas são a anomalia do volume do reservatório Sudeste/Centro-Oeste e a ENA. Incluir o gráfico de Feature Importance do XGBoost (Top 15 variáveis).

A Suavização Mensal (Regra de Negócio): Como a bandeira é mensal, o modelo aplica a Moda intra-mensal para transformar predições diárias em uma predição única por mês, eliminando flutuações espúrias.

Limitações: Risco de Concept Drift (transição energética para eólica/solar), componente discricionário do CMSE na definição de bandeiras, e necessidade de retreinamento periódico.

Conclusão: O modelo binário de XGBoost prova-se eficaz como ferramenta de mitigação de risco financeiro no mercado de energia, entregando métricas sólidas sem contaminação temporal."
