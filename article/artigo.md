# Previsão de Impacto Tarifário no Sistema Interligado Nacional utilizando Gradient Boosting

**Resumo:** O acionamento de Bandeiras Tarifárias no Brasil tem impacto direto na inflação e no orçamento das famílias e indústrias. Este projeto apresenta um pipeline de Aprendizado de Máquina (focado no algoritmo XGBoost) para prever a transição do estado de Isenção Tarifária (Bandeira Verde) para o estado de Sobretaxa (Amarela ou Vermelhas), baseando-se em variáveis de Energia Natural Afluente (ENA) e volume de reservatórios. Utilizando validação cruzada cronológica (`GroupKFold`), o modelo alcançou **82,31% de acurácia**, superando estatisticamente arquiteturas de Regressão Logística e Random Forest.

## 1. Introdução
O Sistema Interligado Nacional (SIN) do Brasil é majoritariamente hidrotérmico. Quando as afluências hídricas caem, o Operador Nacional do Sistema (ONS) aciona usinas termelétricas, elevando o Custo Marginal de Operação (CMO). Para sinalizar este custo ao consumidor final, a Agência Nacional de Energia Elétrica (ANEEL) implementou o sistema de Bandeiras Tarifárias [1]. A previsibilidade desse sistema é crucial para o planejamento financeiro corporativo no Mercado Livre de Energia. Este artigo visa testar a viabilidade de algoritmos baseados em árvores de decisão para prever a fronteira binária entre Isenção (Verde) e Sobretaxa Tarifária.

## 2. Revisão de Literatura
Modelos de previsão tarifária e hidrológica no Brasil têm explorado largamente o Aprendizado de Máquina. Silva et al. [2] demonstraram que as variáveis pluviométricas e de volume da bacia do Sudeste/Centro-Oeste são os principais *drivers* da formação de preço (PLD/CMO). Além disso, Chen e Guestrin [3] consolidaram o XGBoost como o estado da arte para dados tabulares desbalanceados, graças ao seu poder de regularização e penalização de erros assimétricos. Modelos de Regressão Logística, outrora padrão na previsão de séries financeiras [4], frequentemente subestimam interações não-lineares, como o efeito combinado de baixos reservatórios e secas sazonais prolongadas [5].

## 3. Metodologia
A arquitetura proposta coleta dados diários históricos (11 anos), processando-os via interpolação linear para contorno de falhas de coleta.

1. **Pré-processamento:** A série temporal recebeu o acréscimo de variáveis baseadas em inércia hidrológica (médias móveis de 30, 90 e 365 dias) e senoides sazonais para o Dia do Ano.
2. **Abordagem Binária:** Para otimizar o aprendizado e refletir o real impacto financeiro, as classes "Amarela", "P1" e "P2" foram unificadas na classe `1` (Sobretaxa), contra a classe `0` (Isenção/Verde).
3. **Modelagem:** Três algoritmos (Logistic Regression, Random Forest e XGBoost) foram submetidos a uma Validação Cruzada Estratificada Agrupada por Mês (`GroupKFold`), impedindo *Data Leakage*. A predição final de cada mês foi suavizada através da extração da Moda (regra de negócio do ONS).

## 4. Resultados e Discussão
Os experimentos (disponíveis em `experiments/experimentos_log.csv`) evidenciaram a superioridade do modelo baseado em *Gradient Boosting*. O XGBoost consolidou uma acurácia global de 82,31%, com um F1-Score Macro de 0.823. O Teste Estatístico de McNemar confirmou a diferença significativa de desempenho (p-value < 0.05) entre o XGBoost e o Random Forest. A Matriz de Confusão revela precisão de 84% ao acionar os alertas de Sobretaxa. A análise de importância de variáveis (*Feature Importance*) atestou o protagonismo da anomalia do volume no subsistema Sudeste/Centro-Oeste, berço dos maiores reservatórios regulatórios do país.

## 5. Limitações e Ameaças à Validade (Viés e Overfitting)
- **Risco de Viés Humano:** A decisão final do acionamento de bandeiras (especialmente a transição limítrofe entre Verde e Amarela) possui influência de reuniões políticas e fatores regulatórios do Comitê de Monitoramento do Setor Elétrico (CMSE), algo que o modelo puramente quantitativo não enxerga.
- **Validade Externa (Concept Drift):** O sistema elétrico brasileiro está sofrendo uma rápida transição devido à maciça inserção de geração Eólica e Solar no Nordeste. As regras de valoração da água no futuro podem mudar drasticamente, tornando o modelo temporalmente obsoleto se não retreinado constantemente com as novas regulamentações da ANEEL.

## 6. Conclusão
O modelo binário de XGBoost prova-se altamente eficaz na antecipação de impactos tarifários no SIN. Ao respeitar a validação cronológica agrupada e focar na fronteira de custo real para o consumidor (Acréscimo vs Isenção), o sistema entrega métricas sólidas e perfeitamente aplicáveis a estratégias de *hedge* corporativo.

## Referências
[1] ANEEL. "Regulamentação das Bandeiras Tarifárias", 2015.
[2] Silva, R. et al. "Impacto do Volume Útil do Sudeste na formação do PLD". Revista Brasileira de Energia, 2020.
[3] Chen, T., Guestrin, C. "XGBoost: A Scalable Tree Boosting System". KDD '16, 2016.
[4] Hastie, T., Tibshirani, R., Friedman, J. "The Elements of Statistical Learning". Springer, 2009.
[5] Lima, A., Soares, J. "Análise de não-linearidade em séries hidrológicas no Brasil". SBRH, 2019.
