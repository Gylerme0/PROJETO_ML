import pandas as pd
import numpy as np
import os
import datetime
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import mode
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
EXP_DIR = os.path.join(BASE_DIR, 'experiments')
os.makedirs(EXP_DIR, exist_ok=True)

SEED = 42

def train_and_evaluate(clf, name, X, y, grupos, df_ref):
    gkf = GroupKFold(n_splits=5)
    y_pred_diario = np.zeros(len(y))
    acc_folds = []
    f1_folds = []
    
    for tr_i, te_i in gkf.split(X, y, grupos):
        X_tr, X_te = X[tr_i], X[te_i]
        y_tr, y_te = y[tr_i], y[te_i]
        
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        y_pred_diario[te_i] = pred
        
        acc_folds.append(accuracy_score(y_te, pred))
        f1_folds.append(f1_score(y_te, pred, average='macro'))

    df_ref['Pred_Diaria'] = y_pred_diario
    df_ref['Pred_Suavizada'] = df_ref.groupby('DataRef')['Pred_Diaria'].transform(lambda x: mode(x, keepdims=True).mode[0])
    y_pred_final = df_ref['Pred_Suavizada'].values
    
    acc_final = accuracy_score(y, y_pred_final)
    f1_final = f1_score(y, y_pred_final, average='macro')
    
    return {
        'model_name': name,
        'params': str(clf.get_params()),
        'cv_acc_mean': np.mean(acc_folds),
        'cv_acc_std': np.std(acc_folds),
        'cv_f1_mean': np.mean(f1_folds),
        'cv_f1_std': np.std(f1_folds),
        'final_smooth_acc': acc_final,
        'final_smooth_f1': f1_final,
        'y_pred_raw': y_pred_diario
    }

def main():
    print("Carregando dados processados...")
    df = pd.read_parquet(os.path.join(PROC_DIR, 'dados_features.parquet'))
    
    feature_cols = [c for c in df.columns if c not in ['Data', 'DataRef', 'Target_Financeiro']]
    X = df[feature_cols].values
    y = df['Target_Financeiro'].values
    grupos = df['DataRef'].astype(str).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    modelos = [
        ("Regressão Logística", LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced')),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=SEED, class_weight='balanced', n_jobs=-1)),
        ("XGBoost", xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=SEED))
    ]

    log_records = []
    preds_dict = {}

    for name, clf in modelos:
        print(f"Treinando {name}...")
        res = train_and_evaluate(clf, name, X, y, grupos, df.copy())
        
        log_records.append({
            'Data': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Modelo': name,
            'Hiperparametros': res['params'],
            'CV_Acc_Mean': round(res['cv_acc_mean'], 4),
            'CV_Acc_Std': round(res['cv_acc_std'], 4),
            'CV_F1_Mean': round(res['cv_f1_mean'], 4),
            'Smooth_Acc': round(res['final_smooth_acc'], 4)
        })
        preds_dict[name] = res['y_pred_raw']
        
        # Salvar o modelo
        joblib.dump(clf, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))

    # Salvar log de experimentos
    log_df = pd.DataFrame(log_records)
    log_file = os.path.join(EXP_DIR, 'experimentos_log.csv')
    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)
    
    print(f"Experimentos registrados em {log_file}")
    
    # Salvar predições para teste estatístico posterior
    np.save(os.path.join(EXP_DIR, 'y_true.npy'), y)
    np.save(os.path.join(EXP_DIR, 'preds_lr.npy'), preds_dict['Regressão Logística'])
    np.save(os.path.join(EXP_DIR, 'preds_rf.npy'), preds_dict['Random Forest'])
    np.save(os.path.join(EXP_DIR, 'preds_xgb.npy'), preds_dict['XGBoost'])

if __name__ == "__main__":
    main()
