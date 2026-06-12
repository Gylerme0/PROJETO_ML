# Projeto de Machine Learning - AV2

## Visão Executiva

### Problema, Objetivo e Variável-Alvo
O Sistema Interligado Nacional (SIN) utiliza o sistema de Bandeiras Tarifárias para sinalizar ao consumidor os custos reais da geração de energia. O problema consiste em prever o acionamento de acréscimos tarifários com base em variáveis hidrológicas e climáticas.
- **Objetivo:** Prever a transição do estado de "Isenção Tarifária" (Bandeira Verde) para o estado de "Sobretaxa" (Bandeira Amarela, Vermelha P1 ou Vermelha P2).
- **Variável-Alvo:** `Target_Financeiro` (0 = Isenção, 1 = Sobretaxa).

### Origem dos Dados e Resumo
- **Bandeiras Tarifárias:** Histórico oficial do ONS/ANEEL.
- **Hidrologia e Energia:** Dados de Energia Natural Afluente (ENA) e Média de Longo Termo (MLT) da base ONS.
- **Clima:** Índices pluviométricos (INMET).
- **Resumo:** A base conta com 4.138 amostras diárias, englobando 11 anos históricos de chuvas, níveis de reservatório e anomalias computadas.

### Instruções de Instalação e Execução
O projeto é 100% reproduzível.
1. Instale as dependências com versões fixas:
   ```bash
   pip install -r requirements.txt
   ```
2. Comando único de execução (Pipeline Ponta-a-Ponta):
   ```bash
   python src/data/make_dataset.py && python src/features/build_features.py && python src/visualization/eda.py && python src/models/train_model.py && python src/evaluation/evaluate_model.py
   ```

### Resumo dos Resultados e Limitações
- O modelo **XGBoost** atingiu **82,31% de acurácia**, superando estatisticamente os modelos de Random Forest e Regressão Logística (comprovado via Teste de McNemar).
- **Limitações:** O modelo depende fortemente do Volume do SE/CO e da ENA. Mudanças bruscas em políticas governamentais ou regras da ANEEL podem introduzir *"concept drift"* (deslocamento de conceito). A transição sutil entre Bandeira Verde e Amarela permanece altamente discricionária por parte do operador humano.
