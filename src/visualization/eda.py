import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FIG_DIR = os.path.join(BASE_DIR, 'article', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

def run_eda():
    print("Iniciando EDA...")
    df = pd.read_parquet(os.path.join(PROC_DIR, 'dados_features.parquet'))
    
    # 1. Distribuição de Classes
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='Target_Financeiro', hue='Target_Financeiro', palette=['#2ecc71', '#e74c3c'], legend=False)
    plt.title('Distribuição de Classes (Isenção vs Sobretaxa)', fontsize=14)
    plt.xlabel('0: Isenção (Verde) | 1: Sobretaxa (Amarela/Vermelha)')
    plt.ylabel('Dias')
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.savefig(os.path.join(FIG_DIR, '01_distribuicao_classes.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Matriz de Correlação (Top Features vs Target)
    cols_corr = [c for c in df.columns if c not in ['Data', 'DataRef']]
    corr = df[cols_corr].corr(method='spearman')['Target_Financeiro'].sort_values(key=abs, ascending=False)
    top_cols = corr.index[1:11] # Top 10 excluding Target itself
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_cols.tolist() + ['Target_Financeiro']].corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação (Top 10 Variáveis e Target)', fontsize=14)
    plt.savefig(os.path.join(FIG_DIR, '02_matriz_correlacao.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 3. Boxplot: Volume SE/CO por Classe
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Target_Financeiro', y='Vol_SE_CO', hue='Target_Financeiro', palette=['#2ecc71', '#e74c3c'], legend=False)
    plt.title('Volume do Reservatório SE/CO vs Impacto Tarifário', fontsize=14)
    plt.xlabel('0: Isenção | 1: Sobretaxa')
    plt.ylabel('Volume Útil (%)')
    plt.savefig(os.path.join(FIG_DIR, '03_boxplot_volume_seco.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 4. Evolução Temporal
    plt.figure(figsize=(12, 6))
    df_plot = df.set_index('Data')[['Vol_SE_CO', 'Chuva_Sudeste_CO']].resample('ME').mean()
    plt.plot(df_plot.index, df_plot['Vol_SE_CO'], label='Volume SE/CO (%)', color='blue')
    plt.plot(df_plot.index, df_plot['Chuva_Sudeste_CO'], label='Chuva Sudeste/CO (mm)', color='cyan', alpha=0.6)
    plt.axhline(y=30, color='red', linestyle='--', label='Cota de Alerta (30%)')
    plt.title('Evolução Mensal do Volume e Chuvas (Sudeste/CO)', fontsize=14)
    plt.legend()
    plt.ylabel('Valores Médios')
    plt.savefig(os.path.join(FIG_DIR, '04_evolucao_temporal.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 5. Dispersão: ENA vs Volume colorido pelo Target
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ENA_SE_CO_pctMLT', y='Vol_SE_CO', hue='Target_Financeiro', palette=['#2ecc71', '#e74c3c'], alpha=0.5, s=20)
    plt.title('Fronteira de Decisão: ENA vs Volume (SE/CO)', fontsize=14)
    plt.xlabel('ENA SE/CO (% MLT)')
    plt.ylabel('Volume SE/CO (%)')
    plt.legend(title='0: Verde | 1: Sobretaxa')
    plt.savefig(os.path.join(FIG_DIR, '05_dispersao_fronteira.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print("5 Gráficos gerados com sucesso em article/figures/")

if __name__ == "__main__":
    run_eda()
