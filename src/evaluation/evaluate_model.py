import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statsmodels.stats.contingency_tables import mcnemar
import xgboost as xgb
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP_DIR = os.path.join(BASE_DIR, 'experiments')
FIG_DIR = os.path.join(BASE_DIR, 'article', 'figures')
MODELS_DIR = os.path.join(BASE_DIR, 'src', 'models')
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')

NOMES_FINANCEIROS = ['Isenção (Verde)', 'Sobretaxa']

def evaluate():
    print("Executando Avaliação e Testes Estatísticos...")
    
    y_true = np.load(os.path.join(EXP_DIR, 'y_true.npy'))
    y_rf = np.load(os.path.join(EXP_DIR, 'preds_rf.npy'))
    y_xgb = np.load(os.path.join(EXP_DIR, 'preds_xgb.npy'))
    
    # McNemar Test: Random Forest vs XGBoost
    # Construir tabela de contingência
    # a: RF acerta, XGB acerta | b: RF acerta, XGB erra
    # c: RF erra, XGB acerta   | d: RF erra, XGB erra
    
    rf_correct = (y_rf == y_true)
    xgb_correct = (y_xgb == y_true)
    
    a = np.sum(rf_correct & xgb_correct)
    b = np.sum(rf_correct & ~xgb_correct)
    c = np.sum(~rf_correct & xgb_correct)
    d = np.sum(~rf_correct & ~xgb_correct)
    
    contingency_table = [[a, b], [c, d]]
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    print("\nTeste Estatístico (McNemar) - Random Forest vs XGBoost:")
    print(f"p-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("Conclusão: Existe diferença estatística significativa entre os modelos.")
    else:
        print("Conclusão: Não há diferença estatística significativa.")
        
    with open(os.path.join(EXP_DIR, 'estatistica_mcnemar.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Teste de McNemar (Random Forest vs XGBoost)\n")
        f.write(f"Tabela de contingência: {contingency_table}\n")
        f.write(f"p-value: {result.pvalue:.4e}\n")

    # Feature Importance do XGBoost
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'XGBoost.pkl'))
    df = pd.read_parquet(os.path.join(PROC_DIR, 'dados_features.parquet'))
    feature_cols = [c for c in df.columns if c not in ['Data', 'DataRef', 'Target_Financeiro']]
    
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[-15:] # Top 15
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importance - XGBoost (Top 15)', fontsize=14)
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig(os.path.join(FIG_DIR, '06_feature_importance.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Avaliação final concluída. Gráficos em article/figures/.")

if __name__ == "__main__":
    evaluate()
