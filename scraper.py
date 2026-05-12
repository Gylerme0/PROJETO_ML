import os
import glob
import sqlite3
import pandas as pd

print("Iniciando a Engenharia de Dados local...")

# ==============================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E BANCO DE DADOS
# ==============================================================================
# Cole aqui os caminhos da sua máquina!
pasta_hidrologico = r'C:\Users\guilh\Desktop\PROJETO_ML\dados_hidrologicos'
arquivo_bandeiras = r'C:\Users\guilh\Desktop\PROJETO_ML\bandeira-tarifaria-acionamento.csv'

conn = sqlite3.connect('base_energia.db')

# # ==============================================================================
# 2. CARREGANDO OS DADOS DA ANEEL (BANDEIRAS)
# ==============================================================================
print("Processando histórico de Bandeiras Tarifárias (Novo Formato)...")
try:
    # Lendo o arquivo com tratamento para o formato brasileiro
    # sep=None com engine='python' detecta automaticamente se é vírgula, ponto e vírgula ou tabulação
    df_bandeiras = pd.read_csv(arquivo_bandeiras, sep=None, engine='python', 
                               encoding='latin1', decimal=',')
    
    # Garantindo que a data de competência seja lida como data
    df_bandeiras['DatCompetencia'] = pd.to_datetime(df_bandeiras['DatCompetencia'])
    
    # Salvando no SQLite
    df_bandeiras.to_sql('tb_bandeiras', conn, if_exists='replace', index=False)
    
    print(f"Tabela 'tb_bandeiras' atualizada com {len(df_bandeiras)} meses de histórico.")

except Exception as e:
    print(f"Erro ao processar o arquivo de bandeiras: {e}")
# ==============================================================================
# 3. CARREGANDO OS DADOS DO ONS (HIDROLÓGICO 2015-2026)
# ==============================================================================
print("Processando arquivos hidrológicos anuais...")

arquivos_ons = glob.glob(os.path.join(pasta_hidrologico, '*.csv'))
lista_df_ons = []

for arquivo in arquivos_ons:
    ano = os.path.basename(arquivo)
    print(f"Lendo e limpando: {ano}")
    try:
        # Lemos de forma crua, sem tentar adivinhar milhares ou decimais ainda
        df_temp = pd.read_csv(arquivo, sep=';', encoding='utf-8')
        
        colunas_uteis = ['din_instante', 'nom_subsistema', 'val_volumeutilcon']
        
        # Verifica se as colunas esperadas existem neste arquivo anual
        if all(coluna in df_temp.columns for coluna in colunas_uteis):
            df_temp = df_temp[colunas_uteis].copy()
            
            # TRATAMENTO DE TEXTO PARA NÚMERO
            df_temp['val_volumeutilcon'] = df_temp['val_volumeutilcon'].astype(str).str.replace(',', '.')
            df_temp['val_volumeutilcon'] = pd.to_numeric(df_temp['val_volumeutilcon'], errors='coerce')
            
            # REMOVENDO NULLS (Usinas a Fio d'água)
            df_temp.dropna(subset=['val_volumeutilcon'], inplace=True)
            
            # =================================================================
            # NOVO: TRATAMENTO DE OUTLIERS EXTREMOS (A "TRAVA DA FÍSICA")
            # =================================================================
            # Qualquer coisa abaixo de 0% ou acima de 110% (margem de vertimento) 
            # é erro de sensor. Transformamos em NaN para a média mensal ignorar esses dias quebrados.
            df_temp.loc[(df_temp['val_volumeutilcon'] < 0) | (df_temp['val_volumeutilcon'] > 110), 'val_volumeutilcon'] = pd.NA
            
            # Como geramos novos NaNs para os erros extremos, removemos eles também
            df_temp.dropna(subset=['val_volumeutilcon'], inplace=True)

            # Renomeamos para o nosso padrão da Etapa 2
            df_temp.rename(columns={
                'din_instante': 'data_medicao', 
                'val_volumeutilcon': 'val_volumeutilpercentual'
            }, inplace=True)
            
            lista_df_ons.append(df_temp)
# ... (resto do código) ...
        else:
            print(f"Colunas não encontradas no arquivo {ano}. Pulando...")
            
    except Exception as e:
        print(f"Erro ao ler o arquivo {ano}: {e}")

# ... (Continua com a parte de salvar no SQLite) ...
# ==============================================================================
# 4. SALVANDO NO BANCO DE DADOS
# ==============================================================================
if lista_df_ons:
    df_hidrologico_completo = pd.concat(lista_df_ons, ignore_index=True)
    
    # Tratamento de segurança: Converter para numérico caso o ONS tenha mandado algum texto sujo
    df_hidrologico_completo['val_volumeutilpercentual'] = pd.to_numeric(df_hidrologico_completo['val_volumeutilpercentual'], errors='coerce')
    
    df_hidrologico_completo.to_sql('tb_hidrologico', conn, if_exists='replace', index=False)
    print(f"\nTabela 'tb_hidrologico' criada com {len(df_hidrologico_completo)} registros no SQLite!")

conn.close()
print("Processo finalizado. Pode rodar a Etapa 2 da Análise Exploratória!")