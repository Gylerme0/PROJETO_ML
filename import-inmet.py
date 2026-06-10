import os
import glob
import sqlite3
import pandas as pd

print("Iniciando o processamento em lote dos dados do INMET...")

# ==============================================================================
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E BANCO
# ==============================================================================
# Altere para a pasta onde você descompactou TODOS os CSVs do INMET
pasta_inmet = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'INMET')
conn = sqlite3.connect('base_energia.db')

arquivos_inmet = glob.glob(os.path.join(pasta_inmet, '**', '*.CSV'), recursive=True)
lista_diaria = []

# ==============================================================================
# 2. FUNÇÃO PARA DESCOBRIR O SUBSISTEMA PELO NOME DO ARQUIVO
# ==============================================================================
def mapear_subsistema(nome_arquivo):
    if '_NE_' in nome_arquivo:
        return 'Nordeste'
    elif '_N_' in nome_arquivo:
        return 'Norte'
    elif '_S_' in nome_arquivo:
        return 'Sul'
    elif '_SE_' in nome_arquivo or '_CO_' in nome_arquivo:
        return 'Sudeste/Centro-Oeste'
    else:
        return 'Outro'

# ==============================================================================
# 3. EXTRAÇÃO E TRANSFORMAÇÃO (ETL)
# ==============================================================================
for arquivo in arquivos_inmet:
    nome_arquivo = os.path.basename(arquivo)
    subsistema = mapear_subsistema(nome_arquivo)
    
    if subsistema == 'Outro':
        continue # Pula arquivos fora do padrão

    print(f"Lendo {nome_arquivo} -> Alocando em: {subsistema}")
    
    try:
        # Arquivos do INMET costumam ter um cabeçalho de 8 linhas com metadados.
        # skiprows=8 pula essas informações inúteis e vai direto para a tabela de dados.
        df_temp = pd.read_csv(arquivo, sep=';', encoding='latin1', skiprows=8, decimal=',')
        
        # O INMET muda os nomes das colunas de tempos em tempos. 
        # Geralmente a data está na coluna 0 e a precipitação (chuva) na coluna 2.
        coluna_data = df_temp.columns[0]
        # Procurando a coluna que contém a palavra 'PRECIPITA' (para evitar erros de acentuação)
        coluna_chuva = [col for col in df_temp.columns if 'PRECIPITA' in col.upper()][0]
        
        # Filtra apenas Data e Chuva
        df_temp = df_temp[[coluna_data, coluna_chuva]].copy()
        df_temp.columns = ['Data', 'Chuva_mm']
        
        # Tratamento de formato e remoção de erros de sensor (-9999 é código de erro do INMET)
        df_temp['Data'] = pd.to_datetime(df_temp['Data'])
        df_temp['Chuva_mm'] = pd.to_numeric(df_temp['Chuva_mm'], errors='coerce')
        df_temp.loc[df_temp['Chuva_mm'] < 0, 'Chuva_mm'] = pd.NA
        
        # Como os dados originais são por HORA, agrupamos por DIA somando a chuva
        df_diario = df_temp.groupby('Data')['Chuva_mm'].sum().reset_index()
        
        # Adiciona a marcação do subsistema a qual essa estação pertence
        df_diario['Subsistema'] = subsistema
        
        lista_diaria.append(df_diario)
        
    except Exception as e:
        print(f"Erro ao processar {nome_arquivo}: {e}")

# ==============================================================================
# 4. CARREGAMENTO NO BANCO DE DADOS (AGREGAÇÃO NACIONAL)
# ==============================================================================
if lista_diaria:
    print("\nConsolidando os dados a nível nacional...")
    # Junta todas as estações do Brasil
    df_brasil = pd.concat(lista_diaria, ignore_index=True)
    
    # O PULO DO GATO: Tiramos a MÉDIA de chuva diária por Subsistema.
    # Se temos 100 estações no Sudeste, qual foi a média de chuva na região inteira naquele dia?
    df_consolidado = df_brasil.groupby(['Data', 'Subsistema'])['Chuva_mm'].mean().reset_index()
    
    # Pivotando a tabela para que cada subsistema vire uma coluna (ideal para o Machine Learning)
    df_pivotado = df_consolidado.pivot(index='Data', columns='Subsistema', values='Chuva_mm').reset_index()
    
    # Renomeando as colunas para ficar elegante no banco de dados
    df_pivotado.columns = ['Data_Medicao', 'Chuva_Nordeste', 'Chuva_Norte', 'Chuva_Sudeste_CO', 'Chuva_Sul']
    
    # Salvando no banco de dados SQLite
    df_pivotado.to_sql('tb_clima_inmet', conn, if_exists='replace', index=False)
    print(f"\nSucesso! Tabela 'tb_clima_inmet' criada com os 4 subsistemas nacionais no banco SQLite.")

conn.close()