## Problema mais crítico: o split temporal (Bloco 2)

 O GroupShuffleSplit embaralha meses aleatoriamente. Isso significa que o modelo pode ter sido treinado com dados de 2022 e testado com dados de 2019 — o que não representa o uso real do modelo. Troca por split puramente cronológico:

def split_e_normalizar(df, feature_cols):
    X      = df[feature_cols].values
    y      = df['Target'].astype(int).values
    grupos = df['DataRef'].astype(str).values

    # ── Split cronológico estrito ──────────────────────────────────────────
    meses_unicos = sorted(df['DataRef'].unique())
    corte        = int(len(meses_unicos) * 0.80)
    # Gap de 1 mês para evitar leakage de rolling features
    meses_treino = set(str(m) for m in meses_unicos[:corte - 1])
    meses_teste  = set(str(m) for m in meses_unicos[corte:])

    mask_tr = np.array([g in meses_treino for g in grupos])
    mask_te = np.array([g in meses_teste  for g in grupos])

X_tr, X_te = X[mask_tr], X[mask_te]
    y_tr, y_te = y[mask_tr], y[mask_te]
    g_tr       = grupos[mask_tr]
    # ... resto igual

    Features de domínio elétrico (maior ganho de qualidade)
As features atuais capturam estado pontual, mas as bandeiras seguem a lógica do ano hidrológico (outubro–setembro) e respondem a tendências de meses. Adiciona isso ao final do carregar_dados_diarios():

# ── Features do ano hidrológico ────────────────────────────────────────────
# O ano hídrico começa em outubro; o modelo precisa saber onde está nele
df['MesHidrologico'] = ((df['Data'].dt.month - 10) % 12) + 1  # out=1 ... set=12
df['MesHid_sin'] = np.sin(2 * np.pi * df['MesHidrologico'] / 12)
df['MesHid_cos'] = np.cos(2 * np.pi * df['MesHidrologico'] / 12)

# ── Tendência de longo prazo (6/12 meses) ──────────────────────────────────
for col in [c for c in df.columns if c.startswith('Vol_') and 'roll' not in c and 'tend' not in c]:
    df[f'{col}_roll180d'] = df[col].rolling(180, min_periods=90).mean()
    df[f'{col}_roll365d'] = df[col].rolling(365, min_periods=180).mean()
    # Desvio percentual do volume atual em relação à média anual (anomalia)
    df[f'{col}_anomalia'] = (df[col] - df[f'{col}_roll365d']) / (df[f'{col}_roll365d'] + 1e-6)

# ── Anomalia ENA vs. MLT ────────────────────────────────────────────────────
for col in [c for c in df.columns if 'pctMLT' in c and 'roll' not in c]:
    df[f'{col}_anomalia_60d'] = df[col].rolling(60, min_periods=30).mean() - 100.0
    df[f'{col}_abaixo_mlt']   = (df[col] < 80).astype(int)  # binário: ENA < 80% MLT

# ── Lags do alvo (simulam memória regulatória) ─────────────────────────────
# Atenção: usar apenas os lags MENSAIS para evitar leakage intra-mês
df['Target_lag1m'] = df.groupby(df['Data'].dt.to_period('M'))['Target'].transform('first').shift(30)
df['Target_lag2m'] = df.groupby(df['Data'].dt.to_period('M'))['Target'].transform('first').shift(60)
df['Target_lag3m'] = df.groupby(df['Data'].dt.to_period('M'))['Target'].transform('first').shift(90)


## Os lags de target são a feature mais poderosa para dados de séries temporais regulatórias — a bandeira de janeiro é fortíssimo preditor da de fevereiro.

def treinar_binario(X_tr, y_tr_bin, grupos, nome, n_trials=50):
    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        # Calcula peso automaticamente para a classe minoritária
        n_neg = (y_tr_bin == 0).sum()
        n_pos = (y_tr_bin == 1).sum()
        spw_auto = n_neg / max(n_pos, 1)

        params = {
            # ... parâmetros existentes ...
            'scale_pos_weight': trial.suggest_float('scale_pos_weight',
                                                     spw_auto * 0.5,
                                                     spw_auto * 2.0),
            # Remove SMOTE do loop interno
        }
        scores = []
        for f_tr_i, f_val_i in gkf.split(X_tr, y_tr_bin, grupos):
            Xf_tr, Xf_val = X_tr[f_tr_i], X_tr[f_val_i]
            yf_tr, yf_val = y_tr_bin[f_tr_i], y_tr_bin[f_val_i]
            if len(np.unique(yf_tr)) < 2: continue
            m = xgb.XGBClassifier(**params)
            m.fit(Xf_tr, yf_tr)
            scores.append(f1_score(yf_val, m.predict(Xf_val), average='binary', zero_division=0))
        return np.mean(scores) if scores else 0.0

## Calibração de probabilidade + custo assimétrico (Bloco 5)

A Etapa 2 (Amarela vs. Vermelha) é onde mais se erra. O XGBoost nativo tende a probabilidades mal calibradas em dados desbalanceados. Adiciona antes do threshold tuning:

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import fbeta_score

### Calibra o modelo da Etapa 2 (mais crítica)
m2_cal = CalibratedClassifierCV(m2, method='isotonic', cv='prefit')
m2_cal.fit(X_tr_e2, y_tr_e2)   # fit no mesmo subconjunto

### No tunar_thresholds, usa F2-score (penaliza mais falsos negativos de crise)
f1 = fbeta_score(y_te, pred, beta=2, average='macro', zero_division=0)

## Pós-processamento: suavização temporal
As bandeiras têm inércia regulatória — não mudam toda semana. Um filtro simples de janela elimina predições espúrias:

def suavizar_predicoes(pred_array, janela=7):
    """
    Substitui predições isoladas pela moda da janela ao redor.
    Evita Verde→P2→Verde em dias consecutivos, o que nunca ocorre na realidade.
    """
    from scipy.stats import mode
    pred_suave = pred_array.copy()
    metade = janela // 2
    for i in range(metade, len(pred_array) - metade):
        vizinhos = pred_array[i - metade: i + metade + 1]
        pred_suave[i] = mode(vizinhos, keepdims=True).mode[0]
    return pred_suave

## Ordem de implementação recomendada
Faz as mudanças nessa sequência para medir o impacto isolado de cada uma:

Split cronológico — corrige a métrica de avaliação (você vai ver o F1 real cair um pouco, mas passar a ser honesto)
Features do ano hidrológico + lags do target — ganho mais consistente
scale_pos_weight no lugar do SMOTE externo — simplifica e corrige leakage
Calibração da Etapa 2 + F2-score — melhora especificamente Amarela e P2
Suavização temporal — últimos pontos percentuais


O que eu olharia antes de concluir se está bom:

Qual é o F1 por classe — especialmente Amarela e Verm.P2, que são as mais raras e mais importantes
Qual a matriz de confusão — Verde confundindo com Amarela é aceitável; Verde confundindo com P2 não é
Qual é o baseline mais simples possível — se "repete a bandeira do mês anterior" já acerta 65%, seu modelo precisa superar isso com folga