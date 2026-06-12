import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from pipeline_cascata import carregar_dados_diarios
import xgboost as xgb

warnings.filterwarnings('ignore')

df, feature_cols = carregar_dados_diarios()
X = df[feature_cols].values
y = df['Target'].values
grupos = df['DataRef'].astype(str).values

gkf = GroupKFold(n_splits=5)
y_pred_all = np.zeros(len(y))
y_true_all = np.zeros(len(y))

# Arrays to accumulate
fold = 0
for tr_i, te_i in gkf.split(X, y, grupos):
    fold += 1
    X_tr, X_te = X[tr_i], X[te_i]
    y_tr, y_te = y[tr_i], y[te_i]
    
    # Etapa 1: Normal (0,1) vs Crise (2,3)
    y_tr_e1 = np.where(y_tr >= 2, 1, 0)
    m1 = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=42)
    m1.fit(X_tr, y_tr_e1)
    pred_e1 = m1.predict(X_te)
    
    # Etapa 2: Verde (0) vs Amarela (1)
    mask_norm = y_tr < 2
    y_tr_e2 = y_tr[mask_norm]
    m2 = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
    m2.fit(X_tr[mask_norm], y_tr_e2)
    
    # Etapa 3: P1 (2) vs P2 (3)
    mask_crise = y_tr >= 2
    y_tr_e3 = np.where(y_tr[mask_crise] == 3, 1, 0)
    m3 = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=-1, random_state=42)
    m3.fit(X_tr[mask_crise], y_tr_e3)
    
    # Predicao final
    pred_final = np.zeros(len(X_te))
    
    idx_norm = np.where(pred_e1 == 0)[0]
    if len(idx_norm) > 0:
        pred_norm = m2.predict(X_te[idx_norm])
        pred_final[idx_norm] = pred_norm
        
    idx_crise = np.where(pred_e1 == 1)[0]
    if len(idx_crise) > 0:
        pred_crise = m3.predict(X_te[idx_crise])
        pred_final[idx_crise] = np.where(pred_crise == 1, 3, 2)
        
    y_pred_all[te_i] = pred_final
    y_true_all[te_i] = y_te

print(f"\nNova Cascata Acc: {accuracy_score(y_true_all, y_pred_all):.2%}")
print(classification_report(y_true_all, y_pred_all))

# Smooth
df['pred'] = y_pred_all
df['pred_smooth'] = df.groupby('DataRef')['pred'].transform(lambda x: x.mode()[0])
print(f"\nNova Cascata + Smooth Mensal Acc: {accuracy_score(y, df['pred_smooth']):.2%}")
