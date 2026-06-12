# Roteiro de Apresentação (4 Pessoas — 15 Minutos)

Cada participante terá em torno de 3 a 3,5 minutos de fala, permitindo transições tranquilas.

---

## Pessoa 1: Contextualização e o Problema do Negócio (00:00 — 03:30)

**Abertura:** "Bom dia/boa noite a todos. Nosso projeto aplica Machine Learning para resolver um dos problemas mais críticos e custosos do setor elétrico brasileiro: a previsão do impacto financeiro das Bandeiras Tarifárias na conta de luz."

**O Cenário:** Explicar como funciona o SIN (Sistema Interligado Nacional) e o papel do ONS. Focar na matriz predominantemente hidrelétrica e na dependência das chuvas.

**O Problema:** Quando não chove o suficiente (degradação dos reservatórios), o ONS precisa despachar usinas termelétricas, que são muito mais caras. Isso eleva o CMO (Custo Marginal de Operação) e aciona o mecanismo de Bandeiras Tarifárias da ANEEL — gerando sobretaxas na conta de luz.

**O Objetivo:** Prever se haverá **Isenção Tarifária** (Bandeira Verde) ou **Sobretaxa** (Bandeira Amarela, Vermelha P1 ou Vermelha P2) nos próximos meses. É um problema de **Classificação Binária** orientado ao impacto financeiro real, vital para indústrias e distribuidoras se prepararem economicamente.

---

## Pessoa 2: Os Dados e a Engenharia de Features (03:30 — 07:00)

**As Fontes:** "Para alimentar nosso modelo, não basta olhar para uma única fonte." Explicar a união de dados de três fontes governamentais oficiais:
- **ANEEL:** Histórico de acionamentos de bandeiras (variável-alvo).
- **ONS:** Energia Natural Afluente (ENA em MWmed e % da MLT) e Volume Útil dos Reservatórios por subsistema (SE/CO, Sul, Norte, Nordeste).
- **INMET:** Precipitação diária por região.

**A Escala Diária:** Explicar que, para ganhar amostragem robusta (4.138 dias, 11 anos de histórico), os dados foram expandidos para escala diária, mesmo a bandeira sendo definida mensalmente. Isso permitiu captar variações finas de volume e chuva.

**A Engenharia de Variáveis:** Falar sobre a criação de 45 atributos matemáticos — médias móveis de 30, 90 e 365 dias para capturar a inércia hidrológica, anomalias de volume (quanto abaixo da média anual estamos), senoides sazonais do ano hidrológico (outubro–setembro) e anomalias da ENA em relação à MLT.

**O Desbalanceamento:** Citar que a maioria dos meses é Bandeira Verde. Modelos ingênuos sempre chutam "Verde" e parecem ter alta acurácia — o que é uma ilusão perigosa chamada de "paradoxo da acurácia".

---

## Pessoa 3: Modelagem, Evolução e Algoritmos (07:00 — 10:30)

**A Tentativa Multiclasse (e por que fracassou):** "Inicialmente, tentamos prever a cor exata da bandeira (Verde, Amarela, P1, P2) usando Regressão Logística, SVM e Random Forest. Os resultados foram limitados — Macro F1 de apenas 0.47 — porque a fronteira entre Verde e Amarela é altamente discricionária (depende de decisões políticas do ONS), criando um 'teto de vidro' para a predição multiclasse."

**O Pivô para Impacto Financeiro:** Explicar a decisão de negócio: para o consumidor e para a indústria, o que importa é saber se **vai ter sobretaxa ou não**. Unificamos Amarela, P1 e P2 em uma única classe "Sobretaxa", transformando o problema em classificação binária.

**Os Três Modelos Comparados:**
1. **Regressão Logística (Multinomial):** Modelo baseline com alta interpretabilidade.
2. **Random Forest:** Ensemble de árvores resistente a outliers climáticos.
3. **XGBoost (Gradient Boosting):** Algoritmo estado da arte para dados tabulares, com regularização contra overfitting e capacidade de lidar com desbalanceamento. Foi o **modelo campeão** do projeto.

---

## Pessoa 4: Validação, Resultados e Conclusão (10:30 — 14:00)

**O Perigo do Data Leakage:** Explicar que ao usar dados diários, uma validação cruzada simples (StratifiedKFold) misturava dias do mesmo mês entre treino e teste, fazendo o modelo "trapacear" com 99.9% de acurácia. A solução foi o **GroupKFold cronológico**, que agrupa todos os dias de um mesmo mês no mesmo fold, impedindo contaminação temporal.

**A Suavização Mensal (Regra de Negócio):** Como a bandeira é definida para o mês inteiro, aplicamos a Moda intra-mensal: o modelo analisa os 30 dias e crava a predição pela tendência majoritária.

**Os Resultados Concretos:**
- XGBoost alcançou **82,31% de acurácia global** e **F1-Score Macro de 0.823**.
- Precision de **84%** ao prever Sobretaxa — ou seja, quando o modelo alerta crise, ele acerta 84% das vezes.
- Recall de **81%** para Sobretaxa — captura 81% dos períodos reais de crise.
- O Teste de McNemar confirmou a **superioridade estatística** do XGBoost sobre o Random Forest.

**Feature Importance:** A análise de importância de variáveis revelou que a anomalia do volume dos reservatórios do Sudeste/Centro-Oeste e a ENA são os preditores mais fortes.

**Encerramento:** "Concluímos que o realinhamento do objetivo de Machine Learning com o impacto financeiro real — e a utilização de validação cronológica rigorosa — nos permitiu construir uma arquitetura preditiva madura, com 82% de acurácia sem qualquer contaminação temporal. O modelo funciona como um escudo de risco financeiro para o mercado de energia brasileiro. Obrigado pela atenção."

*(Deixar 1 minuto final de margem para dúvidas dos avaliadores.)*
