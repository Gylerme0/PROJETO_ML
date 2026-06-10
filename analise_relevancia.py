# Análise da relevância de cada variável para prever bandeiras tarifárias
import sqlite3
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import pointbiserialr, kruskal

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'base_energia.db')
PASTA = os.path.join(BASE_DIR, 'graficos')
os.makedirs(PASTA, exist_ok=True)

SEED = 42
NOMES_BANDEIRAS = {0: 'Verde', 1: 'Amarela', 2: 'Verm. P1', 3: 'Verm. P2'}
CORES = {0: '#2ecc71', 1: '#f1c40f', 2: '#e74c3c', 3: '#8b0000'}
MAPA = {'Verde': 0, 'Amarela': 1, 'Vermelha P1': 2, 'Vermelha P2': 3, 'Escassez Hídrica': 3}

# ── Carregar dados ─────────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
df_agua  = pd.read_sql("SELECT data_medicao, nom_subsistema, val_volumeutilpercentual FROM tb_hidrologico", conn)
df_chuva = pd.read_sql("SELECT Data_Medicao, Chuva_Nordeste, Chuva_Norte, Chuva_Sudeste_CO, Chuva_Sul FROM tb_clima_inmet", conn)
df_band  = pd.read_sql("SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras", conn)
conn.close()

# ── Pré-processamento ──────────────────────────────────────────────────────────
df_agua['Data'] = pd.to_datetime(df_agua['data_medicao']).dt.to_period('M').dt.to_timestamp()
df_agua = df_agua[(df_agua['val_volumeutilpercentual'] >= 0) & (df_agua['val_volumeutilpercentual'] <= 110)]

vol = df_agua.groupby(['Data','nom_subsistema'])['val_volumeutilpercentual'].mean().unstack().reset_index()
vol.columns = ['Data','Vol_NE','Vol_Norte','Vol_SE_CO','Vol_Sul']

df_chuva['Data'] = pd.to_datetime(df_chuva['Data_Medicao']).dt.to_period('M').dt.to_timestamp()
chuva = df_chuva.groupby('Data')[['Chuva_Nordeste','Chuva_Norte','Chuva_Sudeste_CO','Chuva_Sul']].sum().reset_index()

df_band['Data'] = pd.to_datetime(df_band['DatCompetencia']).dt.to_period('M').dt.to_timestamp()
df_band['Target'] = df_band['NomBandeiraAcionada'].map(MAPA)
df_band['Bandeira'] = df_band['Target'].map(NOMES_BANDEIRAS)

df = pd.merge(vol, chuva, on='Data').merge(df_band[['Data','Target','Bandeira']], on='Data').dropna()
df.sort_values('Data', inplace=True)

# Lags
for c in ['Vol_SE_CO','Vol_NE','Vol_Sul','Vol_Norte']:
    df[f'{c}_Lag1'] = df[c].shift(1)
    df[f'{c}_Lag2'] = df[c].shift(2)
    df[f'{c}_Delta'] = df[c].diff()
for c in ['Chuva_Nordeste','Chuva_Norte','Chuva_Sudeste_CO','Chuva_Sul']:
    df[f'{c}_Acum2M'] = df[c].rolling(2).sum()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

features = [c for c in df.columns if c not in ['Data','Target','Bandeira']]
X = df[features]
y = df['Target'].astype(int)

# ── Treinar modelo para permutation importance ─────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
sc = StandardScaler()
Xtr = sc.fit_transform(X_tr)
Xte = sc.transform(X_te)

sw = compute_sample_weight('balanced', y_tr)
hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05, random_state=SEED)
hgb.fit(Xtr, y_tr, sample_weight=sw)

perm = permutation_importance(hgb, Xte, y_te, n_repeats=30, random_state=SEED, scoring='f1_macro')
imp_df = pd.DataFrame({'Feature': features, 'Importancia': perm.importances_mean, 'Std': perm.importances_std})
imp_df.sort_values('Importancia', ascending=False, inplace=True)

# Classificar grupo de cada feature
def grupo(f):
    if 'Chuva' in f and 'Acum' in f: return 'Chuva Acumulada (Lag)'
    if 'Chuva' in f: return 'Chuva Mensal'
    if 'Lag' in f or 'Delta' in f: return 'Volume com Lag'
    return 'Volume Corrente'

imp_df['Grupo'] = imp_df['Feature'].apply(grupo)
cores_grupo = {
    'Volume Corrente':       '#3498db',
    'Volume com Lag':        '#2980b9',
    'Chuva Mensal':          '#e67e22',
    'Chuva Acumulada (Lag)': '#e74c3c',
}

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Importância por Permutação (ordenada)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 9))
for _, row in imp_df.iterrows():
    cor = cores_grupo[row['Grupo']]
    ax.barh(row['Feature'], row['Importancia'], xerr=row['Std'],
            color=cor, alpha=0.85, edgecolor='black', linewidth=0.4, capsize=3)

# Linha vertical no zero
ax.axvline(0, color='gray', lw=1, ls='--')

legend_patches = [plt.matplotlib.patches.Patch(color=c, label=l) for l,c in cores_grupo.items()]
ax.legend(handles=legend_patches, title='Grupo de Variável', loc='lower right', fontsize=10)
ax.set_xlabel('Redução no Macro F1 quando a variável é embaralhada\n(valores maiores = variável mais relevante)', fontsize=11)
ax.set_title('Importância por Permutação — Previsão de Bandeiras Tarifárias\n(HistGradientBoosting, 30 repetições)', fontsize=13, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, 'relevancia_variaveis.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Salvo: relevancia_variaveis.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Boxplots das 4 variáveis mais relevantes vs Bandeira
# ══════════════════════════════════════════════════════════════════════════════
top4 = imp_df.head(4)['Feature'].tolist()
ordem = ['Verde', 'Amarela', 'Verm. P1', 'Verm. P2']
cores_box = [CORES[0], CORES[1], CORES[2], CORES[3]]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, feat in enumerate(top4):
    ax = axes[i]
    data_plot = [df[df['Target'] == t][feat].dropna().values for t in [0,1,2,3]]
    bp = ax.boxplot(data_plot, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, cor in zip(bp['boxes'], cores_box):
        patch.set_facecolor(cor)
        patch.set_alpha(0.75)
    ax.set_xticklabels(ordem, fontsize=10)
    ax.set_title(f'{feat}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Bandeira Tarifária')
    ax.grid(True, alpha=0.3)

    # Teste de Kruskal-Wallis (non-parametric ANOVA)
    grupos = [df[df['Target'] == t][feat].dropna().values for t in [0,1,2,3]]
    stat, pval = kruskal(*grupos)
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
    ax.set_title(f'{feat}\nKruskal-Wallis: p={pval:.4f} {sig}', fontsize=10, fontweight='bold')

plt.suptitle('Top 4 Variáveis Mais Relevantes vs Bandeira Tarifária\n(* p<0.05, ** p<0.01, *** p<0.001)', 
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PASTA, 'top4_variaveis_vs_bandeira.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Salvo: top4_variaveis_vs_bandeira.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Correlação de Spearman de TODAS as features com o Target
# ══════════════════════════════════════════════════════════════════════════════
from scipy.stats import spearmanr

corrs = []
for feat in features:
    r, p = spearmanr(df[feat], df['Target'])
    corrs.append({'Feature': feat, 'Spearman_r': r, 'p_value': p, 'Grupo': grupo(feat)})

corr_df = pd.DataFrame(corrs).sort_values('Spearman_r')

fig, ax = plt.subplots(figsize=(11, 9))
for _, row in corr_df.iterrows():
    cor = cores_grupo[row['Grupo']]
    sig_marker = '*' if row['p_value'] < 0.05 else ''
    ax.barh(row['Feature'] + sig_marker, row['Spearman_r'], 
            color=cor, alpha=0.85, edgecolor='black', linewidth=0.4)

ax.axvline(0, color='black', lw=1.5, ls='-')
ax.axvline(0.3, color='gray', lw=1, ls=':', alpha=0.5)
ax.axvline(-0.3, color='gray', lw=1, ls=':', alpha=0.5)

legend_patches = [plt.matplotlib.patches.Patch(color=c, label=l) for l,c in cores_grupo.items()]
ax.legend(handles=legend_patches, title='Grupo', loc='lower right', fontsize=9)
ax.set_xlabel('Correlação de Spearman com a Bandeira (Target)\n(* = estatisticamente significativo, p<0.05)', fontsize=11)
ax.set_title('Correlação de Spearman: Cada Variável vs Bandeira Tarifária\n(negativo = quanto menor o valor, mais grave a bandeira)', 
             fontsize=12, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PASTA, 'correlacao_spearman_vs_target.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Salvo: correlacao_spearman_vs_target.png")

# ══════════════════════════════════════════════════════════════════════════════
# PRINTS DE ANÁLISE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  RANKING DE RELEVÂNCIA — TOP 10 VARIÁVEIS")
print("="*65)
print(f"{'#':<3} {'Variável':<30} {'Importância':<14} {'Grupo'}")
print("-"*65)
for i, (_, row) in enumerate(imp_df.head(10).iterrows(), 1):
    print(f"{i:<3} {row['Feature']:<30} {row['Importancia']:<14.4f} {row['Grupo']}")

print("\n" + "="*65)
print("  CORRELAÇÃO DE SPEARMAN — TOP 10 (valor absoluto)")
print("="*65)
corr_abs = corr_df.copy()
corr_abs['abs_r'] = corr_abs['Spearman_r'].abs()
corr_abs = corr_abs.sort_values('abs_r', ascending=False)
print(f"{'#':<3} {'Variável':<30} {'Spearman r':<14} {'p-value':<12} {'Sig.'}")
print("-"*65)
for i, (_, row) in enumerate(corr_abs.head(10).iterrows(), 1):
    sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else 'ns'))
    print(f"{i:<3} {row['Feature']:<30} {row['Spearman_r']:<14.4f} {row['p_value']:<12.4f} {sig}")

print("\n" + "="*65)
print("  KRUSKAL-WALLIS — PODER DISCRIMINATÓRIO POR VARIÁVEL")
print("="*65)
kw_results = []
for feat in features:
    grupos = [df[df['Target'] == t][feat].dropna().values for t in [0,1,2,3]]
    try:
        stat, pval = kruskal(*grupos)
        kw_results.append({'Feature': feat, 'stat': stat, 'p_value': pval, 'Grupo': grupo(feat)})
    except: pass

kw_df = pd.DataFrame(kw_results).sort_values('stat', ascending=False)
print(f"{'#':<3} {'Variável':<30} {'KW stat':<12} {'p-value':<12} {'Grupo'}")
print("-"*65)
for i, (_, row) in enumerate(kw_df.head(10).iterrows(), 1):
    sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else 'ns'))
    print(f"{i:<3} {row['Feature']:<30} {row['stat']:<12.2f} {row['p_value']:<12.4f} {row['Grupo']} {sig}")

print()
