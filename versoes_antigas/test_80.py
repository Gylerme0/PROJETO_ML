import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import warnings
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from pipeline_cascata import carregar_dados_diarios
import xgboost as xgb

warnings.filterwarnings('ignore')

df, feature_cols = carregar_dados_diarios()
y = df['Target'].values
grupos = df['DataRef'].astype(str).values

print("\n--- Teste 1: Apenas Lags ---")
cols_lags = ['Target_lag1m', 'Target_lag2m', 'Target_lag3m']
X_lags = df[cols_lags].fillna(0).values

gkf = GroupKFold(n_splits=5)
accs = []
for tr_i, te_i in gkf.split(X_lags, y, grupos):
    m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=42)
    m.fit(X_lags[tr_i], y[tr_i])
    accs.append(accuracy_score(y[te_i], m.predict(X_lags[te_i])))
print(f"Lags-only Acc: {np.mean(accs):.2%}")

print("\n--- Teste 2: Lags + ENA + Vol ---")
cols_sel = cols_lags + [c for c in feature_cols if 'ENA' in c or 'Vol' in c]
X_sel = df[cols_sel].fillna(0).values
accs = []
for tr_i, te_i in gkf.split(X_sel, y, grupos):
    m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
    m.fit(X_sel[tr_i], y[tr_i])
    accs.append(accuracy_score(y[te_i], m.predict(X_sel[te_i])))
print(f"Lags+ENA+Vol Acc: {np.mean(accs):.2%}")

print("\n--- Teste 3: Lags + Suavização Temporal ---")
from scipy.stats import mode
accs = []
for tr_i, te_i in gkf.split(X_sel, y, grupos):
    m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
    m.fit(X_sel[tr_i], y[tr_i])
    pred = m.predict(X_sel[te_i])
    
    # Suaviza a predição para o mês inteiro (já que a bandeira é mensal)
    # Pega a moda da predição dentro de cada mês do teste
    df_te = df.iloc[te_i].copy()
    df_te['pred'] = pred
    df_te['pred_smooth'] = df_te.groupby('DataRef')['pred'].transform(lambda x: x.mode()[0])
    
    accs.append(accuracy_score(y[te_i], df_te['pred_smooth']))
print(f"Lags+ENA+Vol + Mensal Smoothing Acc: {np.mean(accs):.2%}")

