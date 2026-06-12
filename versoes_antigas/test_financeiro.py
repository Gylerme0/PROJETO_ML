import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import warnings
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from pipeline_cascata import carregar_dados_diarios
import xgboost as xgb

warnings.filterwarnings('ignore')

df, feature_cols = carregar_dados_diarios()
X = df[feature_cols].values
y = df['Target'].values
grupos = df['DataRef'].astype(str).values

# Simplificação Financeira: 0 (Verde - Sem Acréscimo) vs 1 (Amarela/P1/P2 - Com Acréscimo)
y_bin = np.where(y >= 1, 1, 0)

gkf = GroupKFold(n_splits=5)
accs = []
y_true_all = []
y_pred_all = []

for tr_i, te_i in gkf.split(X, y_bin, grupos):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
    m.fit(X[tr_i], y_bin[tr_i])
    pred = m.predict(X[te_i])
    accs.append(accuracy_score(y_bin[te_i], pred))
    y_true_all.extend(y_bin[te_i])
    y_pred_all.extend(pred)

print(f"\n--- Verde vs Acréscimo Tarifário ---")
print(f"Accuracy: {np.mean(accs):.2%}")
print(classification_report(y_true_all, y_pred_all, target_names=["Verde (Sem Taxa)", "Acréscimo (Amarela+)"]))

# Adicionando Smooth Mensal para refletir a regra real
df['pred'] = y_pred_all
df['pred_smooth'] = df.groupby('DataRef')['pred'].transform(lambda x: x.mode()[0])
print(f"Accuracy com Smooth Mensal: {accuracy_score(y_bin, df['pred_smooth']):.2%}")
