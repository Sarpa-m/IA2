# coletor_senado.py
import requests
import pandas as pd
import time
from datetime import datetime
import re # Importa regex para filtragem

# --- Configuração ---
BASE_URL = "https://legis.senado.leg.br/dadosabertos"
HEADERS = {"Accept": "application/json"}
# --------------------

def coletar_votacoes_eficiente(data_inicio, data_fim):
    """
    Coleta todas as votações nominais do Senado em um determinado período
    (formatos AAAA-MM-DD).
    """
    
    print(f"Buscando votações nominais de {data_inicio} a {data_fim}...")
    
    url_endpoint = f"{BASE_URL}/votacao"
 
    params = {
        "dataInicio": data_inicio,
        "dataFim": data_fim,
        "v": 1
    }

    try:
        response = requests.get(url_endpoint, headers=HEADERS, params=params)
        response.raise_for_status() 
        
        lista_votacoes = response.json()

        if not lista_votacoes:
            print(f"Nenhuma votação encontrada no período {data_inicio} a {data_fim}.")
            return None

        print(f"Encontradas {len(lista_votacoes)} votações nominais. Processando...")

        todos_os_votos = []
        n_secretas_puladas = 0 

        for votacao in lista_votacoes:
            
            # 1. Filtro de Votação Secreta
            if 'S' == votacao.get("votacaoSecreta"):
                n_secretas_puladas += 1
                continue 

            # [MODIFICADO] Coleta os metadados da votação
            codigo_votacao = votacao.get("codigoSessaoVotacao")
            data_votacao = votacao.get("dataSessao")
            # Adiciona a descrição e identificação para o filtro de mérito
            descricao_votacao = votacao.get("descricaoVotacao", "")
            identificacao_materia = votacao.get("identificacao", "")
            
            votos_parlamentares = votacao.get("votos", [])

            for voto in votos_parlamentares:
                linha_dataset = {
                    "codigo_votacao": codigo_votacao,
                    "data_votacao": data_votacao,
                    # [NOVO] Campos adicionados para filtragem
                    "descricao_votacao": descricao_votacao,
                    "identificacao_materia": identificacao_materia,
                    "codigo_parlamentar": voto.get("codigoParlamentar"),
                    "nome_parlamentar": voto.get("nomeParlamentar"),
                    "partido_parlamentar": voto.get("siglaPartidoParlamentar"),
                    "uf_parlamentar": voto.get("siglaUFParlamentar"),
                    "sigla_voto": voto.get("siglaVotoParlamentar"),
                    "descricao_voto": voto.get("descricaoVotoParlamentar")
                }
                todos_os_votos.append(linha_dataset)
        
        if n_secretas_puladas > 0:
            print(f"Total de {n_secretas_puladas} votações secretas foram ignoradas.")

        if not todos_os_votos:
            print("Processamento concluído, mas nenhum voto nominal foi extraído.")
            return None

        return pd.DataFrame(todos_os_votos)

    except requests.exceptions.RequestException as e:
        print(f"Erro fatal de conexão ao buscar votações: {e}")
        return None
    except requests.exceptions.JSONDecodeError:
        print("Erro: A resposta da API não foi um JSON válido. Verifique os parâmetros.")
        return None
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")
        return None

# [FUNÇÃO MODIFICADA]
def filtrar_dataset_inteligente(df_total: pd.DataFrame, min_presenca_percent: float = 0.5) -> pd.DataFrame:
    """
    Aplica filtros inteligentes ao dataset completo, conforme metodologia do artigo.
    1. [NOVO] Remove votações procedimentais/regimentais (ruído).
    2. Remove votações unânimes (sem variância).
    3. Remove parlamentares com baixa frequência.
    """
    
    print("\n--- Iniciando Filtragem Inteligente ---")

    # --- 1. [NOVO FILTRO CRÍTICO] Remover Votações Procedimentais ---
    # Esta é a implementação do "passo crítico" do artigo [cite: 49]
    print("Filtrando votações procedimentais (mantendo apenas mérito)...")
    
    # Palavras-chave que indicam votação não-substantiva (regimental/procedural)
    # Esta lista é uma aproximação baseada em "táticas regimentares" 
    PALAVRAS_CHAVE_PROCEDIMENTAIS = [
        'requerimento', 'urgência', 'destaque', 'regime de urgência',
        'votação em globo', 'adiamento', 'redação final', 
        'parecer da comissão', 'questão de ordem', 'procedimento'
    ]
    # Compila a regex (ignore case)
    regex_filtro = re.compile('|'.join(PALAVRAS_CHAVE_PROCEDIMENTAIS), re.IGNORECASE)
    
    # Garante que as colunas existam e não tenham NaNs
    df_total['descricao_votacao'] = df_total['descricao_votacao'].fillna('')
    df_total['identificacao_materia'] = df_total['identificacao_materia'].fillna('')
    
    # Combina os textos para a busca
    df_total['texto_busca'] = df_total['descricao_votacao'] + ' ' + df_total['identificacao_materia']
    
    # Identifica as votações que contêm qualquer uma das palavras-chave
    is_procedimental = df_total['texto_busca'].str.contains(regex_filtro, na=False)
    
    votacoes_procedimentais = df_total[is_procedimental]['codigo_votacao'].unique()
    
    n_votacoes_antes = df_total['codigo_votacao'].nunique()
    
    # Filtra o dataframe, mantendo apenas votações que NÃO são procedimentais
    df_filtrado_substantivo = df_total[~df_total['codigo_votacao'].isin(votacoes_procedimentais)].copy()
    
    n_votacoes_depois = df_filtrado_substantivo['codigo_votacao'].nunique()
    print(f"Votações removidas (procedimentais/ruído): {len(votacoes_procedimentais)} (de {n_votacoes_antes} para {n_votacoes_depois} votações)")

    if df_filtrado_substantivo.empty:
        print("Dataset vazio após remoção de votações procedimentais.")
        return df_filtrado_substantivo
    
    # --- 2. Remover Votações Unânimes ---
    print("Filtrando votações unânimes...")
    
    # Usa o dataframe já filtrado (df_filtrado_substantivo)
    df_decisorios = df_filtrado_substantivo[df_filtrado_substantivo['sigla_voto'].isin(['Sim', 'Não'])]
    
    votos_por_sessao = df_decisorios.groupby('codigo_votacao')['sigla_voto'].nunique()
    votacoes_unanimes = votos_por_sessao[votos_por_sessao == 1].index
    
    n_antes_unanime = df_filtrado_substantivo['codigo_votacao'].nunique()
    df_filtrado = df_filtrado_substantivo[~df_filtrado_substantivo['codigo_votacao'].isin(votacoes_unanimes)].copy()
    n_depois_unanime = df_filtrado['codigo_votacao'].nunique()
    
    print(f"Votações removidas por unanimidade: {len(votacoes_unanimes)} (de {n_antes_unanime} para {n_depois_unanime} votações)")

    if df_filtrado.empty:
        print("Dataset vazio após remoção de votações unânimes.")
        return df_filtrado

    # --- 3. Remover Senadores com baixa participação ---
    print(f"Filtrando senadores com menos de {min_presenca_percent*100}% de participação...")
    
    total_votacoes_unicas = df_filtrado['codigo_votacao'].nunique()
    limite_sessoes = total_votacoes_unicas * min_presenca_percent
    
    sessoes_por_senador = df_filtrado.groupby('codigo_parlamentar')['codigo_votacao'].nunique()
    senadores_ativos = sessoes_por_senador[sessoes_por_senador >= limite_sessoes].index
    
    n_senadores_antes = df_total['codigo_parlamentar'].nunique()
    df_final = df_filtrado[df_filtrado['codigo_parlamentar'].isin(senadores_ativos)]
    n_senadores_depois = df_final['codigo_parlamentar'].nunique()
    
    print(f"Senadores removidos por baixa participação: {n_senadores_antes - n_senadores_depois} (de {n_senadores_antes} para {n_senadores_depois} senadores)")
    print("--- Filtragem Inteligente Concluída ---")
    
    # Limpa as colunas de texto que não são mais necessárias
    df_final = df_final.drop(columns=['descricao_votacao', 'identificacao_materia', 'texto_busca'], errors='ignore')
    
    return df_final

def executar_pipeline_coleta(ano_inicio: int, ano_fim: int):
    """
    Orquestra a coleta de dados por vários anos, concatena e salva o resultado final.
    """
    
    print(f"--- Iniciando Pipeline de Coleta de {ano_inicio} a {ano_fim} ---")
    
    lista_dataframes_anuais = []
    
    for ano in range(ano_inicio, ano_fim + 1):
        data_inicio = f"{ano}-01-01"
        data_fim = f"{ano}-12-31"
        
        df_ano = coletar_votacoes_eficiente(data_inicio, data_fim)
        
        if df_ano is not None and not df_ano.empty:
            lista_dataframes_anuais.append(df_ano)
        
        time.sleep(1) 

    if not lista_dataframes_anuais:
        print("Nenhum dado foi coletado em nenhum dos anos. Encerrando.")
        return

    print(f"\nConsolidando dados de {len(lista_dataframes_anuais)} ano(s)...")
    df_total = pd.concat(lista_dataframes_anuais, ignore_index=True)
    print(f"Total de registros de votos coletados: {len(df_total)}")

    df_filtrado = filtrar_dataset_inteligente(df_total, min_presenca_percent=0.5)

    if df_filtrado.empty:
        print("Nenhum dado restou após a filtragem. Nenhum arquivo será salvo.")
        return

    nome_arquivo = f"dataset_votacoes_senado_{ano_inicio}_a_{ano_fim}_FILTRADO.csv"
    # Salva com encoding correto para Excel
    df_filtrado.to_csv(nome_arquivo, index=False, encoding='utf-8-sig')

    print(f"\n--- Pipeline Concluído ---")
    print(f"Dataset final salvo com sucesso em '{nome_arquivo}'!")
    print(f"Total de registros de votos (linhas) no arquivo final: {len(df_filtrado)}")
    print(f"Total de Senadores únicos: {df_filtrado['codigo_parlamentar'].nunique()}")
    print(f"Total de Votações únicas: {df_filtrado['codigo_votacao'].nunique()}")
    
    print("\nAmostra dos dados finais (primeiras 5 linhas):")
    print(df_filtrado.head())


# --- Execução do Script ---
if __name__ == "__main__":
    
    # Esta parte permite que o script ainda seja executável sozinho
    # para fins de teste, usando um período padrão.
    
    # --- Configuração Padrão (apenas se rodar este script direto) ---
    ANO_INICIO_PADRAO = 2023
    ANO_FIM_PADRAO = 2024
    # -----------------------------------------------
    
    print("--- Executando 'coletor_senado.py' em modo standalone ---")
    executar_pipeline_coleta(ANO_INICIO_PADRAO, ANO_FIM_PADRAO)