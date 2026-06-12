# Guia de Interpretação Gráfica (EDA)

Este documento foi criado para auxiliar na compreensão e leitura dos gráficos gerados na fase de Análise Exploratória de Dados (salvos na pasta `article/figures/`).

### 1. `01_distribuicao_classes.png` (Distribuição de Classes)
- **O que mostra:** A quantidade de dias da nossa série histórica que pertencem ao estado de "Isenção" (Verde) versus "Sobretaxa" (Amarela/Vermelha).
- **Como interpretar:** Demonstra o grau de desbalanceamento do problema original. Mesmo agrupando as classes de crise (Amarela, P1 e P2), a classe 0 (Isenção) ainda tem uma leve maioria. Isso justifica o uso de métricas como o F1-Score, que não se deixam enganar por classes majoritárias.

### 2. `02_matriz_correlacao.png` (Matriz de Correlação)
- **O que mostra:** O coeficiente de correlação de Spearman entre as 10 variáveis matemáticas mais fortes do projeto e o nosso Alvo (`Target_Financeiro`).
- **Como interpretar:** Cores azuis fortes indicam correlação negativa (ex: quanto MAIOR o Volume, MENOR a chance de ter sobretaxa). Cores vermelhas fortes indicam correlação positiva (ex: anomalias hídricas aumentam a chance de crise). Isso prova para a banca que o modelo baseia suas decisões na física da água, e não no acaso.

### 3. `03_boxplot_volume_seco.png` (Boxplot do Volume SE/CO)
- **O que mostra:** A distribuição estatística do Volume do Subsistema Sudeste/Centro-Oeste separado por Bandeira (0 ou 1).
- **Como interpretar:** A linha no meio das caixas é a mediana. Repare como a caixa verde (Isenção) está posicionada muito mais acima (geralmente acima de 40% de volume), enquanto a caixa vermelha (Sobretaxa) fica quase sempre abaixo dos 30%. O gráfico comprova que o volume SE/CO é o divisor de águas da tarifa elétrica brasileira.

### 4. `04_evolucao_temporal.png` (Evolução Mensal do Volume e Chuvas)
- **O que mostra:** Uma linha do tempo comparando o regime de chuvas (linhas azuis claras) e o nível do reservatório (linha azul escuro). A linha pontilhada vermelha é a cota de alerta.
- **Como interpretar:** Demonstra a Inércia Hidrológica. Observe que o pico das chuvas ocorre meses antes do pico do reservatório. A água demora a escoar e a ser armazenada. Quando a linha do reservatório despenca abaixo da cota vermelha (como na crise de 2021), as bandeiras vermelhas são acionadas em sequência.

### 5. `05_dispersao_fronteira.png` (Fronteira de Decisão: ENA vs Volume)
- **O que mostra:** Um gráfico de pontos (dispersão) onde o eixo X é a ENA (água chegando) e o eixo Y é o Volume (água guardada). Cada ponto é um dia. Pontos verdes = Isenção; Vermelhos = Sobretaxa.
- **Como interpretar:** Esse é o "Cérebro do XGBoost". Note como existe um quadrante crítico (embaixo à esquerda: baixa ENA e baixo Volume) dominado por pontos vermelhos. E o quadrante oposto (em cima à direita) dominado por pontos verdes. A pequena zona onde os pontos se misturam é a "Zona de Discricionariedade" onde o ONS toma decisões políticas, e é justamente onde o nosso modelo atua para minimizar riscos.
