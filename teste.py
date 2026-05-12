import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Iniciando Fase de Preparação e Modelagem...\n")

# ==============================================================================
# 1. EXTRAÇÃO E PREPARAÇÃO DOS DADOS
# ==============================================================================
conn = sqlite3.connect('base_energia.db')
query = """
    SELECT data_medicao, nom_subsistema, val_volumeutilpercentual 
    FROM tb_hidrologico
"""
df_agua = pd.read_sql(query, conn)
df_bandeiras = pd.read_sql("SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras", conn)
conn.close()

# Ajuste de Datas e Pivotamento (Normalizando para o primeiro dia do mês)
df_agua['Data'] = pd.to_datetime(df_agua['data_medicao']).dt.to_period('M').dt.to_timestamp()
df_agua_mensal = df_agua.groupby(['Data', 'nom_subsistema'])['val_volumeutilpercentual'].mean().unstack().reset_index()
df_agua_mensal.columns = ['Data', 'Vol_NE', 'Vol_Norte', 'Vol_SE_CO', 'Vol_Sul']

# Corrigindo a conversão de data e o mapeamento das bandeiras
df_bandeiras['Data'] = pd.to_datetime(df_bandeiras['DatCompetencia']).dt.to_period('M').dt.to_timestamp()
mapa_bandeiras = {
    'Verde': 0, 
    'Amarela': 1, 
    'Vermelha P1': 2, 
    'Vermelha P2': 3,
    'Escassez Hídrica': 3  # Agrupando com o nível mais alto
}
df_bandeiras['Target'] = df_bandeiras['NomBandeiraAcionada'].map(mapa_bandeiras)

df_final = pd.merge(df_agua_mensal, df_bandeiras[['Data', 'Target']], on='Data', how='inner')
df_final.dropna(subset=['Target'], inplace=True)

# ==============================================================================
# 2. ENGENHARIA DE FEATURES (LAGGED FEATURES)
# ==============================================================================
# Criando o histórico: Como estava a água 1 mês atrás (Lag 1) e 2 meses atrás (Lag 2)
df_final.sort_values('Data', inplace=True)

for col in ['Vol_SE_CO', 'Vol_NE', 'Vol_Sul', 'Vol_Norte']:
    df_final[f'{col}_Lag1'] = df_final[col].shift(1)
    df_final[f'{col}_Lag2'] = df_final[col].shift(2)

# Removemos as duas primeiras linhas pois não têm histórico (Lag) para elas
df_final.dropna(inplace=True)

# Definindo quem são os Previsores (X) e quem é o Alvo (y)
colunas_features = [c for c in df_final.columns if c not in ['Data', 'Target']]
X = df_final[colunas_features]
y = df_final['Target']

# ==============================================================================
# 3. DIVISÃO TREINO/TESTE E NORMALIZAÇÃO (PREVENINDO DATA LEAKAGE)
# ==============================================================================
# Separamos 30% dos dados para testar o modelo no final. 
# stratify=y garante que a proporção de bandeiras vermelhas não se perca na divisão
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# A regra de ouro da AV2: Normalizar calculando apenas no Treino (fit), 
# e aplicando (transform) no treino e no teste
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Dados Treino: {X_train.shape[0]} meses | Dados Teste: {X_test.shape[0]} meses\n")

# ==============================================================================
# 4. TREINAMENTO DOS MODELOS
# ==============================================================================
# class_weight='balanced' é o truque para problemas desbalanceados! Ele penaliza o modelo
# severamente se ele errar uma Bandeira Vermelha.

# Modelo 1: Regressão Logística (Baseline Interpretável)
modelo_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
modelo_lr.fit(X_train_scaled, y_train)
previsoes_lr = modelo_lr.predict(X_test_scaled)

# Modelo 2: Random Forest (Estado da Arte para Tabelas)
modelo_rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42)
modelo_rf.fit(X_train_scaled, y_train)
previsoes_rf = modelo_rf.predict(X_test_scaled)

# ==============================================================================
# 5. AVALIAÇÃO COM A MÉTRICA CORRETA (MACRO F1-SCORE)
# ==============================================================================
print("--- DESEMPENHO: REGRESSÃO LOGÍSTICA ---")
print(f"Macro F1-Score: {f1_score(y_test, previsoes_lr, average='macro'):.3f}")
print(classification_report(y_test, previsoes_lr, target_names=['Verde', 'Amarela', 'Verm_P1', 'Verm_P2']))

print("\n--- DESEMPENHO: RANDOM FOREST ---")
print(f"Macro F1-Score: {f1_score(y_test, previsoes_rf, average='macro'):.3f}")
print(classification_report(y_test, previsoes_rf, target_names=['Verde', 'Amarela', 'Verm_P1', 'Verm_P2']))

# Bônus: Extraindo a Importância das Features do Random Forest
importancias = pd.Series(modelo_rf.feature_importances_, index=colunas_features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importancias.plot(kind='barh', color='teal')
plt.title('Importância das Variáveis (Random Forest)')
plt.xlabel('Peso da Variável na Decisão da Bandeira')
plt.tight_layout()
plt.show()