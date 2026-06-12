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

# Simplificação: 0 (Verde/Amarela) vs 1 (Vermelha P1/P2)
y_bin = np.where(y >= 2, 1, 0)

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

print(f"Crise vs Normal Acc: {np.mean(accs):.2%}")
print(classification_report(y_true_all, y_pred_all))
