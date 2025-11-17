# coletor_senado.py
import requests
import pandas as pd
import time
from datetime import datetime

# --- Configuração ---

# [CORREÇÃO] URL Base da API. Este endpoint não usa o prefixo /api/v1/
BASE_URL = "https://legis.senado.leg.br/dadosabertos"

# Cabeçalho para informar à API que queremos a resposta em formato JSON
HEADERS = {"Accept": "application/json"}

# Período que queremos analisar
# [CORREÇÃO] A documentação deste endpoint especifica o formato AAAA-MM-DD
DATA_INICIO = "2022-01-01"
DATA_FIM = "2022-12-31"

# NOTA: A documentação informa que o intervalo entre as datas deve ser de no máximo 1 ano.
# --------------------


def coletar_votacoes_eficiente(data_inicio, data_fim):
    """
    Coleta todas as votações nominais do Senado em um determinado período
    usando o endpoint /votacao (que é muito mais eficiente) e salva em um CSV.
    """
    
    print(f"Buscando votações nominais de {data_inicio} a {data_fim}...")
    
    # [CORREÇÃO] Endpoint correto, conforme sua documentação
    url_endpoint = f"{BASE_URL}/votacao"
 
    params = {
        "dataInicio": data_inicio,
        "dataFim": data_fim,
        "v": 1  # Incluindo o parâmetro de versão 'v=1' visto no exemplo
    }

    try:
        # Passo 1: Fazer uma ÚNICA chamada para obter todas as votações e seus votos
        response = requests.get(url_endpoint, headers=HEADERS, params=params)
        response.raise_for_status() # Verifica se houve erro HTTP (4xx ou 5xx)
        
        # [CORREÇÃO] A resposta deste endpoint é uma lista direta [...]
        lista_votacoes = response.json()

        if not lista_votacoes:
            print("Nenhuma votação encontrada no período.")
            return

        print(f"Encontradas {len(lista_votacoes)} votações nominais. Processando...")

        # Lista que armazenará todos os registros (linhas) do nosso dataset
        todos_os_votos = []

        # Passo 2: Iterar sobre a lista de votações
        for votacao in lista_votacoes:
            
            # Extrai os dados principais da votação
            codigo_votacao = votacao.get("codigoSessaoVotacao")
            data_votacao = votacao.get("dataSessao")
            descricao_votacao = votacao.get("descricaoVotacao")
            identificacao_materia = votacao.get("identificacao")
            ementa = votacao.get("ementa")

            # [CORREÇÃO] Acessa a lista de votos nominais dentro do objeto da votação
            votos_parlamentares = votacao.get("votos", [])

            # Passo 3: Iterar sobre os votos de CADA votação
            for voto in votos_parlamentares:
                
                # Monta um dicionário (que representa uma linha do dataset)
                linha_dataset = {
                    "codigo_votacao": codigo_votacao,
                    "data_votacao": data_votacao,
                  #  "descricao_votacao": descricao_votacao,
                  #  "identificacao_materia": identificacao_materia,
                  #  "ementa_materia": ementa,
                    "codigo_parlamentar": voto.get("codigoParlamentar"),
                    "nome_parlamentar": voto.get("nomeParlamentar"),
                    "partido_parlamentar": voto.get("siglaPartidoParlamentar"),
                    "uf_parlamentar": voto.get("siglaUFParlamentar"),
                    # 'siglaVotoParlamentar' é o voto bruto (Ex: 'Sim', 'Nao', 'AP', 'P-NRV')
                    "sigla_voto": voto.get("siglaVotoParlamentar"),
                    # 'descricaoVotoParlamentar' dá mais detalhes (Ex: 'Atividade parlamentar')
                    "descricao_voto": voto.get("descricaoVotoParlamentar")
                }
                todos_os_votos.append(linha_dataset)

        if not todos_os_votos:
            print("Processamento concluído, mas nenhum voto nominal foi extraído.")
            return

        print("\nProcessamento de votos nominais concluído.")

        # Passo 4: Criar o DataFrame (a tabela) com todos os dados coletados
        df = pd.DataFrame(todos_os_votos)

        # Passo 5: Salvar o DataFrame em um arquivo CSV
        nome_arquivo = f"dataset_votacoes_senado_{data_inicio}_a_{data_fim}.csv"
        df.to_csv(nome_arquivo, index=False, encoding='utf-8-sig')

        print(f"\nDataset salvo com sucesso em '{nome_arquivo}'!")
        print("Amostra dos dados (primeiras 5 linhas):")
        print(df.head())

    except requests.exceptions.RequestException as e:
        print(f"Erro fatal de conexão ao buscar votações: {e}")
    except requests.exceptions.JSONDecodeError:
        print("Erro: A resposta da API não foi um JSON válido. Verifique os parâmetros.")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")

# --- Execução do Script ---
if __name__ == "__main__":
    coletar_votacoes_eficiente(DATA_INICIO, DATA_FIM)