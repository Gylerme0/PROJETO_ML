# Acompanhamento — AV2 Machine Learning

## Tema do Projeto
**"Previsão de Bandeiras Tarifárias e Avaliação de Impacto Financeiro:
Um Modelo de Classificação Baseado em Dados Hidrometeorológicos"**

## Checklist de Entrega (Mapeado pelas Fases do Prof. Mateus)

### Fase 1: Dados e Exploração (15%)
- [x] Problema definido: Classificação multiclasse (4 bandeiras)
- [x] Dataset com n≥100 amostras (136 meses reais de 2015 a 2026)
- [x] Fontes oficiais: ANEEL, ONS (AWS), INMET
- [x] 7 gráficos de EDA gerados e salvos em `graficos/`
- [x] Desbalanceamento documentado (49% Verde vs 17% Vermelha P2)

### Fase 2: Preparação e Features
- [x] Tratamento de dados faltantes (outliers 0-110%, usinas fio d'água)
- [x] Engenharia de Features: 17 variáveis (Lag1, Lag2, Delta, Carga)
- [x] Divisão Treino/Teste: 70/30 com seed=42 e stratify
- [x] Normalização Z-Score SEM Data Leakage

### Fase 3: Modelagem (20%)
- [x] Modelo 1: Regressão Logística (Multinomial) — Macro F1 = 0.471
- [x] Modelo 2: Random Forest (300 árvores) — Macro F1 = 0.446
- [x] Modelo 3: SVM kernel RBF — Macro F1 = 0.432
- [x] Hiperparâmetros registrados para cada modelo

### Fase 4: Validação Rigorosa (25% — Maior Peso)
- [x] Validação Cruzada Estratificada (5-Fold) nos dados de treino
- [x] Métricas corretas: F1-Score Macro e Weighted (NÃO Acurácia)
- [x] 3 Matrizes de Confusão visuais (gráfico 08)
- [x] Teste de McNemar: p-value = 1.00 (modelos sem diferença significativa)
- [x] Relatório exportado em `resultados/relatorio_metricas.txt`

### Fase 5: Interpretação e Insights (20%)
- [x] Feature Importance do Random Forest (gráfico 09)
- [x] Top 3 variáveis identificadas
- [x] Impacto financeiro calculado por bandeira
- [x] Limitações do modelo documentadas
- [x] Modelo recomendado: Regressão Logística (parcimônia)

### Fase 6: Documentação e Apresentação (20%)
- [x] Código Python limpo, modular e super-comentado
- [x] Relatório técnico (esboço em `relatorio_av2.md`)
- [x] Apresentação oral (roteiro de slides em `slides_av2.md`)

## Estrutura de Arquivos do Projeto
```
PROJETO_ML/
├── scraper.py          — ETL: coleta dados e salva no SQLite
├── main.py             — Pipeline completo (ponto de entrada)
├── etapa1_eda.py       — Análise Exploratória de Dados (7 gráficos)
├── etapa2_modelagem.py — Preparação + Modelagem + Validação + Insights
├── base_energia.db     — Banco SQLite com dados limpos
├── bandeira-tarifaria-acionamento.csv — Dados ANEEL
├── CARGA_MENSAL.parquet — Carga de energia ONS
├── dados_hidrologicos/ — 12 CSVs do ONS (2015-2026)
├── graficos/           — 9 gráficos gerados automaticamente
└── resultados/         — Relatório de métricas exportado
```

## Como Executar
```bash
# 1. (Só precisa fazer uma vez) Gerar o banco de dados
python scraper.py

# 2. Executar o pipeline completo de ML
python main.py
```
