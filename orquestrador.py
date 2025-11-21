import os
import coletor_senado               # Importa seu script de coleta
import analise_votacoes_kmeans      # Importa seu script de análise
import pandas as pd                 # Para as configurações do pandas

# ===================================================================
# --- CONFIGURAÇÃO PRINCIPAL ---
#
# Defina o intervalo de anos desejado APENAS AQUI.
#
ANO_INICIO = 2019
ANO_FIM = 2023


#
# ===================================================================

def executar_orquestrador():
    """
    Verifica se os dados existem. Se não, executa o coletor.
    Depois, executa a análise.
    """
    
    print("="*80)
    print("INICIANDO ORQUESTRADOR DO PIPELINE DE ANÁLISE")
    print(f"Período configurado: {ANO_INICIO} a {ANO_FIM}")
    print("="*80)

    # 1. Definir nome do arquivo padrão
    nome_arquivo = f"dataset_votacoes_senado_{ANO_INICIO}_a_{ANO_FIM}_FILTRADO.csv"
    
    # 2. Verificar se o arquivo existe
    if not os.path.exists(nome_arquivo):
        print(f"\n[FASE 1] Dataset '{nome_arquivo}' não encontrado.")
        print("--- Executando o Coletor de Dados (isso pode levar um tempo) ---")
        
        # Chama a função principal do coletor
        coletor_senado.executar_pipeline_coleta(ANO_INICIO, ANO_FIM)
        
        print("--- Coleta de Dados Concluída ---")
    else:
        print(f"\n[FASE 1] Dataset '{nome_arquivo}' já existe.")
        print("--- Pulando a Coleta de Dados ---")

    # 3. Executar a análise
    print(f"\n[FASE 2] Executando a Análise K-Means no arquivo '{nome_arquivo}'...")
    
    # Configura o pandas (bom ter aqui, já que é o script principal)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    
    # Chama a função principal da análise, passando o nome do arquivo
    # (Vamos modificar 'analise_votacoes_kmeans.py' para aceitar isso)
    analise_votacoes_kmeans.executar_analise_pipeline(nome_arquivo)
    
    print("\n" + "="*80)
    print("ORQUESTRAÇÃO CONCLUÍDA")
    print("="*80)

if __name__ == "__main__":
    executar_orquestrador()