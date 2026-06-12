# Dicionário de Dados (`dados_features.parquet`)

Este documento descreve as variáveis da base processada utilizada no treinamento do modelo.

### 1. Metadados e Target
- **Data (datetime):** Dia exato da medição.
- **DataRef (str):** Mês de referência no formato "YYYY-MM" para o `GroupKFold`.
- **Target_Financeiro (int):** Variável-alvo Binária. `0` significa Isenção (Bandeira Verde) e `1` significa Sobretaxa (Bandeira Amarela, Vermelha P1 ou Vermelha P2).

### 2. Variáveis de Volume (Reservatórios)
- **Vol_NE (float):** Percentual do volume útil no Nordeste.
- **Vol_Norte (float):** Percentual do volume útil no Norte.
- **Vol_SE_CO (float):** Percentual do volume útil no Sudeste/Centro-Oeste.
- **Vol_Sul (float):** Percentual do volume útil no Sul.

### 3. Variáveis Climáticas (INMET)
- **Chuva_Nordeste (float):** Pluviometria diária em milímetros.
- **Chuva_Norte (float):** Pluviometria diária em milímetros.
- **Chuva_Sudeste_CO (float):** Pluviometria diária em milímetros.
- **Chuva_Sul (float):** Pluviometria diária em milímetros.

### 4. Variáveis de Energia (ENA)
- **ENA_[Sub]_MWmed (float):** Energia Natural Afluente Bruta em Megawatt-médio por subsistema.
- **ENA_[Sub]_pctMLT (float):** Energia Natural Afluente Bruta como percentual da Média de Longo Termo (100% = chuvas dentro do esperado).

### 5. Engenharia de Recursos (Feature Engineering)
- **Sazonalidade (`_sin`, `_cos`):** DiaDoAno e MesHidrologico transformados via funções trigonométricas para captar ciclicidade sem descontinuidades entre Dez/Jan.
- **Inércia (`_roll30d`, `_roll90d`, `_roll365d`):** Média móvel dos volumes e percentuais MLT nos últimos N dias, representando o acúmulo hidrológico.
- **Anomalias (`_anomalia_1ano`, `_anomalia_60d`):** Diferença percentual entre o valor atual e o comportamento de longo prazo. Indica choques hídricos (secas repentinas).
