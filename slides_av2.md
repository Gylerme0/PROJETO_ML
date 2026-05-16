# Roteiro para Apresentação Oral — Previsão de Bandeiras Tarifárias (AV2)
*Duração sugerida: 10 a 15 minutos.*

## Slide 1: Título e Apresentação
- **Título:** Previsão de Bandeiras Tarifárias e Avaliação de Impacto Financeiro.
- **Subtítulo:** Um Modelo de Classificação Baseado em Dados Hidrometeorológicos do SIN.
- **Aluno:** Guilherme
- **Professor:** Mateus Silva
- *Fala sugerida:* "Bom dia a todos. Meu projeto visa antecipar quando a conta de luz vai ficar mais cara, usando Inteligência Artificial para analisar o nível das águas no Brasil inteiro."

## Slide 2: O Problema de Negócio (Contexto)
- **Bullet Points:**
  - Forte dependência nacional da matriz hídrica.
  - Baixo volume nas represas = uso de termoelétricas (queima de gás/carvão).
  - Termoelétricas são caras. Para repassar o custo, a ANEEL aciona Bandeiras.
- **Objetivo do Trabalho:** Desenvolver um algoritmo capaz de "prever" qual bandeira será acionada analisando apenas o volume dos quatro reservatórios do Brasil e o consumo do mês.

## Slide 3: Coleta de Dados e Qualidade
- **Mencionar as Fontes:** ANEEL (histórico de bandeiras) e ONS (nível de águas e carga).
- **Tratamento de Dados (ETL):** 
  - Tivemos que limpar mais de 400 mil linhas de medição diária.
  - Sensores da base pública tinham erros absurdos (ex: reservatórios com "700% de água"). Cortamos outliers fora da faixa física de 0% a 110%.
- **O Desbalanceamento:** (Mostrar o Gráfico 01 aqui - `01_distribuicao_bandeiras.png`).
  - Metade do tempo ficamos no Verde. Apenas 17% do tempo é crise brava.
  - Isso forçou o modelo a usar pesos (`class_weight='balanced'`). Acurácia aqui seria um erro estatístico fatal.

## Slide 4: Engenharia de Variáveis (Features)
- *Conceito Principal:* A inércia hídrica. 
- *Explicação:* A água que chove hoje não vai direto pro motor da hidrelétrica. Por isso, criamos colunas olhando para o passado: "Qual era a represa há 1 mês?" (Lag1) e "Há 2 meses?" (Lag2).
- Essa técnica transformou a capacidade do nosso modelo.

## Slide 5: Evitando o "Data Leakage"
- *O rigor técnico:* Separamos 70% da base para Treino e 30% para Teste.
- *Explicação:* O Z-Score (StandardScaler) só olhou a média e desvio do grupo de treino. Se olhássemos para tudo antes de separar, estaríamos "espiando o gabarito" e o modelo seria fraudulento.

## Slide 6: Modelagem (Os 3 Algoritmos)
- Testamos as 3 abordagens:
  1. **Regressão Logística Multinomial** (simples e transparente)
  2. **Random Forest** (poderoso, agrupa centenas de árvores de decisão)
  3. **SVM com RBF** (excelente para desenhar fronteiras em curvas onde a linha reta não dá conta).

## Slide 7: Resultados e Validação Cruzada (A Hora da Verdade)
- *Estratégia:* Usamos Validação Cruzada de 5 partições e medimos pelo **F1-Score Macro** (a métrica que se recusa a ignorar as classes raras, como a Bandeira Vermelha P2).
- **Resultados de Teste:**
  - Regressão Logística: F1 = 0.471
  - Random Forest: F1 = 0.446
  - SVM: F1 = 0.432

## Slide 8: O Teste Estatístico (McNemar)
- Será que a Logística ganhou "por sorte"?
- Aplicamos o rigoroso **Teste de McNemar** comparando os acertos da Logística contra o Random Forest.
- O p-value deu alto (1.000). Estatisticamente, deu um "empate técnico".
- *Decisão técnica:* Pela Navalha de Ockham, sempre que houver empate de desempenho, a gente escolhe a máquina mais simples. A Logística venceu.
- *(Mostrar aqui o Gráfico de Matrizes de Confusão para provar os acertos - `08_matrizes_confusao.png`)*

## Slide 9: Insights de Negócio e Interpretação
- *(Mostrar o Gráfico Feature Importance - `09_feature_importance.png`)*
- Descobrimos que o Sul e o Norte têm exercido enorme pressão na equação recente das bandeiras.
- **Impacto na conta:** Uma Bandeira P2 encarece, em média, mais de R$ 11,80 na conta do consumidor comum, escalando para milhares de reais no caso de indústrias. Prever com 1 mês de antecedência vale ouro para negociações no Mercado Livre.

## Slide 10: Limitações e Conclusão
- **Limitações:** Temos "só" cerca de 136 meses na história moderna das bandeiras. O modelo apanha para prever "Escassez", porque ocorreu em menos de 10 meses na história (evento raro). Fatores políticos de Brasília também não são previstos matematicamente.
- **Fechamento:** O objetivo acadêmico de cobrir todas as frentes de ML – limpeza, engenharia contra Data Leakage, tuning e validação rigorosa com teste estatístico – foi atingido, comprovando a eficácia técnica do projeto.
- "Agradeço a atenção e abro para perguntas do professor."
