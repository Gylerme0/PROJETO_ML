# =============================================================================
# ETAPA 1 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# =============================================================================
# Projeto: Previsão de Bandeiras Tarifárias com Machine Learning
# Aluno: Guilherme
# Disciplina: Machine Learning — AV2
#
# OBJETIVO DESTA ETAPA:
#   Gerar pelo menos 5 visualizações gráficas (exigência da AV2, Fase 1 - 15%)
#   para entender o comportamento dos dados antes de treinar os modelos.
#
# O QUE ESTE SCRIPT FAZ:
#   1. Conecta ao banco SQLite (base_energia.db) gerado pelo scraper.py
#   2. Extrai os dados hidrológicos (nível dos reservatórios) e bandeiras (ANEEL)
#   3. Cruza as tabelas por data (merge)
#   4. Gera 7 gráficos informativos salvos na pasta graficos/
# =============================================================================

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid", font_scale=1.1)
PASTA_GRAFICOS = os.path.join(os.path.dirname(__file__), 'graficos')
os.makedirs(PASTA_GRAFICOS, exist_ok=True)


def carregar_dados():
    """
    Carrega os dados do banco SQLite e faz o cruzamento (merge) entre
    a tabela de reservatórios e a tabela de bandeiras.
    
    Retorna:
        df_final (DataFrame): Tabela unificada com volumes mensais por
                              subsistema e a bandeira acionada naquele mês.
    """
    db_path = os.path.join(os.path.dirname(__file__), 'base_energia.db')
    conn = sqlite3.connect(db_path)

    # --- Dados Hidrológicos (ONS) ---
    # Cada linha = 1 dia de 1 reservatório, com o volume útil (%)
    df_agua = pd.read_sql(
        "SELECT data_medicao, nom_subsistema, val_volumeutilpercentual FROM tb_hidrologico",
        conn
    )

    # --- Dados de Bandeiras (ANEEL) ---
    # Cada linha = 1 mês com a bandeira acionada naquele período
    df_bandeiras = pd.read_sql(
        "SELECT DatCompetencia, NomBandeiraAcionada FROM tb_bandeiras",
        conn
    )
    conn.close()

    # =====================================================================
    # PREPARAÇÃO: Transformar dados diários em médias mensais
    # =====================================================================
    # O ONS fornece dados DIÁRIOS de cada reservatório.
    # A ANEEL fornece a bandeira MENSAL.
    # Precisamos agregar os dados diários em média mensal para cruzar.
    df_agua['Data'] = pd.to_datetime(df_agua['data_medicao']).dt.to_period('M').dt.to_timestamp()
    
    # Pivotamento: cada subsistema vira uma coluna com a média do mês
    df_agua_mensal = (
        df_agua.groupby(['Data', 'nom_subsistema'])['val_volumeutilpercentual']
        .mean()
        .unstack()
        .reset_index()
    )
    df_agua_mensal.columns = ['Data', 'Vol_NE', 'Vol_Norte', 'Vol_SE_CO', 'Vol_Sul']

    # Preparar bandeiras: converter data e mapear nomes para números
    df_bandeiras['Data'] = pd.to_datetime(df_bandeiras['DatCompetencia']).dt.to_period('M').dt.to_timestamp()
    
    # MAPEAMENTO DA VARIÁVEL-ALVO (Target):
    # 0 = Verde (sem custo extra), 1 = Amarela, 2 = Vermelha P1, 3 = Vermelha P2/Escassez
    mapa = {
        'Verde': 0,
        'Amarela': 1,
        'Vermelha P1': 2,
        'Vermelha P2': 3,
        'Escassez Hídrica': 3  # Escassez = pior cenário = mesmo nível de Vermelha P2
    }
    df_bandeiras['Target'] = df_bandeiras['NomBandeiraAcionada'].map(mapa)
    df_bandeiras['NomBandeira'] = df_bandeiras['NomBandeiraAcionada']

    # MERGE: cruzar reservatórios com bandeiras pelo mês
    df_final = pd.merge(
        df_agua_mensal,
        df_bandeiras[['Data', 'Target', 'NomBandeira']],
        on='Data', how='inner'
    )
    df_final.dropna(subset=['Target'], inplace=True)
    df_final.sort_values('Data', inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    print(f"✅ Base unificada: {len(df_final)} meses (de {df_final['Data'].min().strftime('%Y-%m')} a {df_final['Data'].max().strftime('%Y-%m')})")
    print(f"   Subsistemas: Sudeste/CO, Nordeste, Sul, Norte")
    print(f"   Distribuição das Bandeiras:")
    nomes = {0: 'Verde', 1: 'Amarela', 2: 'Vermelha P1', 3: 'Vermelha P2'}
    for k, v in df_final['Target'].value_counts().sort_index().items():
        print(f"     {nomes.get(k, k)}: {v} meses ({v/len(df_final)*100:.1f}%)")

    return df_final


def gerar_graficos_eda(df):
    """
    Gera 7 gráficos de Análise Exploratória (EDA) exigidos pela AV2.
    Todos são salvos na pasta graficos/.
    """
    nomes_bandeiras = {0: 'Verde', 1: 'Amarela', 2: 'Verm. P1', 3: 'Verm. P2'}
    cores_bandeiras = {0: '#2ecc71', 1: '#f1c40f', 2: '#e74c3c', 3: '#8b0000'}

    # =====================================================================
    # GRÁFICO 1: Distribuição das Bandeiras (Variável-Alvo)
    # =====================================================================
    # POR QUE ESTE GRÁFICO É IMPORTANTE?
    # Ele revela o DESBALANCEAMENTO das classes. A maioria dos meses é
    # "Verde", tornando difícil para o modelo aprender a prever as crises.
    # O professor Mateus alertou: usar Acurácia com dados desbalanceados
    # é um ERRO FATAL. Este gráfico justifica o uso de F1-Score.
    fig, ax = plt.subplots(figsize=(8, 5))
    contagem = df['Target'].value_counts().sort_index()
    cores = [cores_bandeiras[i] for i in contagem.index]
    bars = ax.bar([nomes_bandeiras[i] for i in contagem.index], contagem.values, color=cores, edgecolor='black')
    for bar, val in zip(bars, contagem.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title('Gráfico 1 — Distribuição das Bandeiras Tarifárias (Variável-Alvo)', fontsize=13)
    ax.set_xlabel('Bandeira Acionada')
    ax.set_ylabel('Quantidade de Meses')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, '01_distribuicao_bandeiras.png'), dpi=150)
    plt.close()
    print("  📊 Gráfico 1 salvo: Distribuição das Bandeiras")

    # =====================================================================
    # GRÁFICO 2: Matriz de Correlação
    # =====================================================================
    # POR QUE? A correlação mostra QUAIS variáveis têm relação com o Target.
    # Se Vol_SE_CO tem correlação -0.8 com Target, significa que quando o
    # Sudeste seca, a bandeira sobe (correlação negativa forte).
    fig, ax = plt.subplots(figsize=(8, 6))
    cols = ['Vol_SE_CO', 'Vol_NE', 'Vol_Sul', 'Vol_Norte', 'Target']
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax,
                vmin=-1, vmax=1, linewidths=0.5)
    ax.set_title('Gráfico 2 — Matriz de Correlação: Reservatórios vs Bandeira', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, '02_correlacao_regioes.png'), dpi=150)
    plt.close()
    print("  📊 Gráfico 2 salvo: Matriz de Correlação")

    # =====================================================================
    # GRÁFICO 3: Boxplot — Nível do Nordeste (Sobradinho) vs Bandeira
    # =====================================================================
    # POR QUE? Mostra como o nível do reservatório de Sobradinho se comporta
    # em cada tipo de bandeira. Se a mediana cai conforme a bandeira piora,
    # confirma que o nível do NE é um bom preditor.
    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df.copy()
    df_plot['Bandeira'] = df_plot['Target'].map(nomes_bandeiras)
    ordem = ['Verde', 'Amarela', 'Verm. P1', 'Verm. P2']
    sns.boxplot(data=df_plot, x='Bandeira', y='Vol_NE', order=ordem,
                palette=[cores_bandeiras[i] for i in range(4)], ax=ax)
    ax.set_title('Gráfico 3 — Volume do Nordeste (Sobradinho) por Bandeira', fontsize=13)
    ax.set_ylabel('Volume Útil NE (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, '03_boxplot_ne_vs_bandeira.png'), dpi=150)
    plt.close()
    print("  📊 Gráfico 3 salvo: Boxplot NE vs Bandeira")

    # =====================================================================
    # GRÁFICO 4: Evolução Temporal dos Reservatórios SE/CO vs NE
    # =====================================================================
    # POR QUE? Séries temporais mostram os ciclos de seca e cheia.
    # Permite identificar visualmente os períodos de crise hídrica
    # (2015-2016 e 2021) que coincidiram com bandeiras vermelhas.
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['Data'], df['Vol_SE_CO'], label='Sudeste/CO (70% da capacidade)', color='#3498db', linewidth=2)
    ax.plot(df['Data'], df['Vol_NE'], label='Nordeste / Sobradinho (18%)', color='#e67e22', linewidth=2)
    ax.plot(df['Data'], df['Vol_Sul'], label='Sul (7%)', color='#27ae60', linewidth=1, alpha=0.6)
    ax.plot(df['Data'], df['Vol_Norte'], label='Norte (5%)', color='#9b59b6', linewidth=1, alpha=0.6)
    # Colorir fundo conforme bandeira
    for _, row in df.iterrows():
        cor = cores_bandeiras.get(row['Target'], '#ffffff')
        ax.axvspan(row['Data'] - pd.Timedelta(days=15),
                   row['Data'] + pd.Timedelta(days=15),
                   alpha=0.15, color=cor, linewidth=0)
    ax.set_title('Gráfico 4 — Evolução Histórica dos Reservatórios (2015–2026)', fontsize=13)
    ax.set_ylabel('Volume Útil (%)')
    ax.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, '04_evolucao_historica.png'), dpi=150)
    plt.close()
    print("  📊 Gráfico 4 salvo: Evolução Temporal")

    # =====================================================================
    # GRÁFICO 5: Boxplot de Volumes por Subsistema (Outliers)
    # =====================================================================
    # POR QUE? Identifica outliers (valores extremos) nos dados.
    # Pontos fora dos "bigodes" do boxplot são anomalias que precisam
    # ser tratadas na preparação dos dados.
    fig, ax = plt.subplots(figsize=(8, 5))
    df_melt = df[['Vol_SE_CO', 'Vol_NE', 'Vol_Sul', 'Vol_Norte']].melt(var_name='Subsistema', value_name='Volume (%)')
    sns.boxplot(data=df_melt, x='Subsistema', y='Volume (%)', palette='Set2', ax=ax)
    ax.set_title('Gráfico 5 — Distribuição de Volume por Subsistema (Outliers)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, '05_outliers_volume.png'), dpi=150)
    plt.close()
    print("  📊 Gráfico 5 salvo: Outliers por Subsistema")

    # =====================================================================
    # GRÁFICO 6: Dispersão SE/CO vs NE (Fronteira de Decisão)
    # =====================================================================
    # POR QUE? Este gráfico mostra como os pontos se agrupam no espaço.
    # Se bandeiras vermelhas ficam no canto inferior-esquerdo (ambos secos),
    # isso prova que o modelo pode aprender a separar as classes.
    fig, ax = plt.subplots(figsize=(8, 6))
    for t in sorted(df['Target'].unique()):
        subset = df[df['Target'] == t]
        ax.scatter(subset['Vol_SE_CO'], subset['Vol_NE'], label=nomes_bandeiras[t],
                   color=cores_bandeiras[t], s=60, edgecolors='black', linewidth=0.5, alpha=0.8)
    ax.set_title('Gráfico 6 — Fronteira de Decisão: Sudeste vs Nordeste', fontsize=13)
    ax.set_xlabel('Volume Sudeste/CO (%)')
    ax.set_ylabel('Volume Nordeste (%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, '06_dispersao_fronteira.png'), dpi=150)
    plt.close()
    print("  📊 Gráfico 6 salvo: Dispersão / Fronteira")

    # =====================================================================
    # GRÁFICO 7: Timeline das Bandeiras
    # =====================================================================
    fig, ax = plt.subplots(figsize=(14, 3))
    for _, row in df.iterrows():
        ax.barh(0, width=30, left=row['Data'], color=cores_bandeiras[row['Target']], edgecolor='none')
    ax.set_yticks([])
    ax.set_title('Gráfico 7 — Linha do Tempo das Bandeiras Tarifárias (2015–2026)', fontsize=13)
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=cores_bandeiras[i], label=nomes_bandeiras[i]) for i in range(4)]
    ax.legend(handles=legend_patches, loc='upper right', ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_GRAFICOS, '07_timeline_bandeiras.png'), dpi=150)
    plt.close()
    print("  📊 Gráfico 7 salvo: Timeline")

    print(f"\n✅ Todos os gráficos salvos em: {PASTA_GRAFICOS}/")


# =============================================================================
# EXECUÇÃO
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("  ETAPA 1: ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("=" * 70)
    df = carregar_dados()
    print("\nGerando gráficos da EDA...")
    gerar_graficos_eda(df)
