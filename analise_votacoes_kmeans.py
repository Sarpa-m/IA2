# -*- coding: utf-8 -*-
"""
Implementação do Pipeline de Análise de Votações com K-Means e Aderência Ideológica

Este script aplica a metodologia descrita para agrupar senadores com base
em seus padrões de votação e calcular uma métrica de Aderência Ideológica (AI).

Metodologia:
A. Coleta e Estruturação dos Dados:
   - Carregamento dos dados (aqui, usamos um exemplo estático).
   - Mapeamento de votos (Sim: +1, Não: -1, Outros: 0).
   - Criação da matriz senador-votação.
B. Modelo de Agrupamento (K-Means):
   - Uso do K-Means com inicialização 'k-means++'.
C. Validação e Otimização (K ótimo):
   - Método do Cotovelo (Inércia/WCSS).
   - Análise de Silhueta.
D. Algoritmo de Aderência Ideológica (AI):
   - Definição do "Voto Esperado" (binarização/trinarização do centróide).
   - Cálculo da AI via Similaridade de Cosseno.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# --- A. Coleta e Estruturação dos Dados ---

def carregar_dados_csv(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo CSV.
    
    NOTA: Esta função é usada para ler de um arquivo real.
    """
    print(f"Carregando dados de {caminho_arquivo}...")
    try:
        # Tenta carregar o CSV. 
        # Ajuste os parâmetros (ex: sep=';') se o seu CSV usar um separador diferente.
        df = pd.read_csv(caminho_arquivo)
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{caminho_arquivo}'")
        print("Por favor, verifique o caminho e o nome do arquivo.")
        # Retorna um DataFrame vazio para evitar que o script quebre
        return pd.DataFrame() 
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        return pd.DataFrame()

def carregar_dados_exemplo() -> pd.DataFrame:
    """
    Carrega um conjunto de dados de exemplo.
    
    NOTA: Substitua o conteúdo desta função pela sua lógica de
    coleta de dados da API do Senado para análise real.
    """
    # Dados de exemplo fornecidos
    dados_csv = """codigo_votacao,data_votacao,codigo_parlamentar,nome_parlamentar,partido_parlamentar,uf_parlamentar,sigla_voto,descricao_voto
6818,2024-02-20T00:00:00,5672,Alan Rick,UNIÃO,AC,Sim,
6818,2024-02-20T00:00:00,5982,Alessandro Vieira,MDB,SE,AP,Atividade parlamentar
6818,2024-02-20T00:00:00,5967,Angelo Coronel,PSD,BA,Sim,
6818,2024-02-20T00:00:00,6009,Astronauta Marcos Pontes,PL,SP,Sim,
6818,2024-02-20T00:00:00,6350,Augusta Brito,PT,CE,Sim,
6819,2024-02-21T00:00:00,5672,Alan Rick,UNIÃO,AC,Sim,
6819,2024-02-21T00:00:00,5982,Alessandro Vieira,MDB,SE,Não,
6819,2024-02-21T00:00:00,5967,Angelo Coronel,PSD,BA,Sim,
6819,2024-02-21T00:00:00,6009,Astronauta Marcos Pontes,PL,SP,Não,
6819,2024-02-21T00:00:00,6350,Augusta Brito,PT,CE,Sim,
6820,2024-02-22T00:00:00,5672,Alan Rick,UNIÃO,AC,Abstenção,
6820,2024-02-22T00:00:00,5982,Alessandro Vieira,MDB,SE,Não,
6820,2024-02-22T00:00:00,5967,Angelo Coronel,PSD,BA,Sim,
6820,2024-02-22T00:00:00,6009,Astronauta Marcos Pontes,PL,SP,Não,
6820,2024-02-22T00:00:00,6350,Augusta Brito,PT,CE,Sim,
"""
    # Adicionei mais algumas votações fictícias para tornar o exemplo menos trivial
    
    return pd.read_csv(StringIO(dados_csv))

def mapear_votos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte votos categóricos em numéricos (Sim: +1, Não: -1, Outros: 0).
    """
    mapa_votos = {
        'Sim': 1,
        'Não': -1
    }
    
    # Mapeia Sim/Não e preenche todos os outros (Abstenção, AP, Ausência, etc.) com 0
    df['voto_numerico'] = df['sigla_voto'].map(mapa_votos).fillna(0)
    return df

def criar_matriz_senador_votacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma o DataFrame de votos na matriz senador-votação.
    Linhas: Senadores
    Colunas: Votações
    Valores: Voto numérico (1, -1, 0)
    """
    # Usar 'codigo_parlamentar' ou 'nome_parlamentar' como índice
    # 'codigo_parlamentar' é mais robusto para evitar nomes duplicados
    
    # Filtragem de votações substantivas (conforme metodologia [1])
    # Aqui, precisaríamos de uma lógica para identificar e filtrar votações
    # procedimentais. Como não temos essa informação, usamos todas.
    # Ex: df = df[df['tipo_votacao'] == 'substantiva'] 
    
    print("Criando matriz senador-votação...")
    matriz = df.pivot_table(
        index='codigo_parlamentar', # Pode usar 'nome_parlamentar'
        columns='codigo_votacao',
        values='voto_numerico',
        fill_value=0 # Ausência/Não participou da votação = 0
    )
    
    # Adiciona nomes para referência futura
    mapa_nomes = df.drop_duplicates('codigo_parlamentar').set_index('codigo_parlamentar')['nome_parlamentar']
    matriz['nome_parlamentar'] = matriz.index.map(mapa_nomes)
    matriz = matriz.set_index('nome_parlamentar', append=True).swaplevel(0, 1)
    
    return matriz

def pre_processar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa o pipeline de pré-processamento.
    """
    df = mapear_votos(df)
    matriz = criar_matriz_senador_votacao(df)
    return matriz

# --- C. Validação e Otimização do Modelo ---

def plotar_validacao_k(k_range: range, inertia_scores: list, silhouette_scores: list):
    """
    Plota os gráficos do Método do Cotovelo e da Análise de Silhueta.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Método do Cotovelo (Elbow Method)
    ax1.plot(k_range, inertia_scores, 'bo-', markerfacecolor='r')
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Inércia (WCSS)')
    ax1.set_title('Método do Cotovelo (Elbow)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Análise de Silhueta (Silhouette Analysis)
    ax2.plot(k_range, silhouette_scores, 'bo-', markerfacecolor='r')
    ax2.set_xlabel('Número de Clusters (K)')
    ax2.set_ylabel('Coeficiente de Silhueta Médio')
    ax2.set_title('Análise de Silhueta')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Validação do Número Ótimo de Clusters (K)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salva o gráfico em um arquivo
    plt.savefig("validacao_k_otimo.png")
    print("\nGráfico de validação 'validacao_k_otimo.png' salvo.")
    # plt.show() # Descomente se quiser exibir o gráfico interativamente

def encontrar_k_otimo(matrix: pd.DataFrame, max_k: int = 10) -> int:
    """
    Testa diferentes valores de K e plota os gráficos de validação.
    Retorna o K que maximiza o coeficiente de silhueta.
    """
    print(f"Iniciando validação para K... (max_k={max_k})")
    
    # Garante que max_k não seja maior que o número de amostras - 1
    # (Necessário para silhouette_score)
    n_samples = matrix.shape[0]
    if n_samples <= max_k:
        max_k = n_samples - 1 
        print(f"Aviso: max_k ajustado para {max_k} (n_samples - 1) para permitir cálculo da silhueta.")

    if max_k < 2:
        print("Aviso: Dados insuficientes para clusterização (menos de 2 amostras). Retornando K=1.")
        return 1

    k_range = range(2, max_k + 1)
    inertia_scores = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(matrix)
        
        # Inércia (WCSS) para o Método do Cotovelo
        inertia_scores.append(kmeans.inertia_)
        
        # Coeficiente de Silhueta
        labels = kmeans.labels_
        score = silhouette_score(matrix, labels)
        silhouette_scores.append(score)
        print(f"  K={k}: Inércia={kmeans.inertia_:.2f}, Silhueta={score:.4f}")

    # Plotar os gráficos
    plotar_validacao_k(k_range, inertia_scores, silhouette_scores)
    
    # Escolher o K que maximiza a silhueta
    if not silhouette_scores:
        return 1
        
    k_otimo = k_range[np.argmax(silhouette_scores)]
    print(f"\nK ótimo identificado (maximizando silhueta): {k_otimo}")
    
    return k_otimo

# --- D. Algoritmo de Classificação para Aderência Ideológica ---

def definir_voto_esperado(centroids: np.ndarray, tau: float = 0.05) -> np.ndarray:
    """
    Trinariza os centróides (μj) para definir o Voto Esperado (ˆvj).
    
    ˆv(i)j =
        +1 se μ(i)j > τ
        -1 se μ(i)j < -τ
         0 se -τ <= μ(i)j <= τ
    """
    
    # np.where é eficiente para esta lógica condicional
    votos_esperados = np.where(
        centroids > tau, 1, np.where(centroids < -tau, -1, 0)
    )
    return votos_esperados

def calcular_aderencia_ideologica(matrix: pd.DataFrame, k_otimo: int) -> pd.DataFrame:
    """
    Executa o K-Means final e calcula a Aderência Ideológica (AI)
    para cada senador.
    """
    print(f"\nCalculando K-Means final com K={k_otimo}...")
    
    kmeans_final = KMeans(n_clusters=k_otimo, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(matrix)
    
    # Centróides (μ)
    centroids = kmeans_final.cluster_centers_
    
    # 1. Definição do Voto Esperado (ˆv)
    votos_esperados_clusters = definir_voto_esperado(centroids, tau=0.05)
    
    print("Calculando Aderência Ideológica (AI) via Similaridade de Cosseno...")
    
    ai_scores = []
    senadores = matrix.index
    
    # 2. Cálculo da Aderência Ideológica (AI)
    for i, (nome_parlamentar, codigo_parlamentar) in enumerate(senadores):
        # Vetor de voto real do senador (xi)
        vetor_senador_x = matrix.iloc[i].values.reshape(1, -1)
        
        # Cluster ao qual o senador foi atribuído
        cluster_atribuido = clusters[i]
        
        # Vetor de voto esperado do cluster ideal (ˆvideal)
        vetor_esperado_v = votos_esperados_clusters[cluster_atribuido].reshape(1, -1)
        
        # Cálculo da Similaridade de Cosseno
        # AIi = Similaridade(xi, ˆvideal)
        ai_score = cosine_similarity(vetor_senador_x, vetor_esperado_v)[0][0]
        
        ai_scores.append({
            'codigo_parlamentar': codigo_parlamentar,
            'nome_parlamentar': nome_parlamentar,
            'cluster': cluster_atribuido,
            'aderencia_ideologica (AI)': ai_score
        })

    return pd.DataFrame(ai_scores).set_index('nome_parlamentar')

# --- Função Principal ---

def main():
    """
    Executa o pipeline completo de análise.
    """
    # A. Coleta e Estruturação
    
    # --- MODIFICAÇÃO: Como importar um CSV ---
    # 1. Comente a linha que usa dados de exemplo:
    # df_bruto = carregar_dados_exemplo()
    
    # 2. Descomente a linha abaixo e coloque o nome do seu arquivo CSV:
    nome_do_seu_arquivo = "dataset_votacoes_senado_2024-01-01_a_2024-12-31.csv" # <-- COLOQUE O NOME DO SEU ARQUIVO AQUI
    df_bruto = carregar_dados_csv(nome_do_seu_arquivo)
    
    # Verifica se os dados foram carregados com sucesso
    if df_bruto.empty:
        print("Não foi possível carregar os dados. Encerrando o script.")
        return
    # --- FIM DA MODIFICAÇÃO ---
    
    # NOTA: Filtragem de Votações
    # A metodologia [1] sugere filtrar votações procedimentais.
    # Esta etapa deve ser inserida aqui, antes de criar a matriz.
    # df_filtrado = filtrar_votacoes_substantivas(df_bruto)
    # matriz_votacoes = pre_processar(df_filtrado)
    
    matriz_votacoes = pre_processar(df_bruto)
    
    # Extrai apenas os dados numéricos para o K-Means
    # (O índice multi-nível 'nome_parlamentar', 'codigo_parlamentar' é mantido)
    # Linha 307 (Corrigida)
    matriz_numerica = matriz_votacoes.loc[:, matriz_votacoes.columns.astype(str).str.isnumeric()]
    
    print("\n--- Matriz Senador-Votação (Amostra) ---")
    print(matriz_numerica)
    print("-" * 50)
    
    # C. Validação e Otimização
    # (Limitamos o max_k para o exemplo, em dados reais use 10 ou 15)
    k_otimo = encontrar_k_otimo(matriz_numerica, max_k=10)
    
    # Para o dataset de exemplo, k_otimo será trivial (provavelmente 2 ou 3)
    # Vamos usar os dados de exemplo para encontrar o k
    n_samples = matriz_numerica.shape[0]
    #k_otimo = encontrar_k_otimo(matriz_numerica, max_k=n_samples-1)

    if k_otimo <= 1:
        print("\nNão foi possível realizar o agrupamento (K ótimo <= 1).")
        print("Verifique se há dados suficientes e variância nos votos.")
        return

    # D. Cálculo da Aderência Ideológica
    df_resultados = calcular_aderencia_ideologica(matriz_numerica, k_otimo)
    
    print("\n--- Resultados Finais: Aderência Ideológica ---")
    print(df_resultados.sort_values(by=['cluster', 'aderencia_ideologica (AI)'], ascending=[True, False]))
    print("-" * 50)
    
    # Salvar resultados
    df_resultados.to_csv("resultados_aderencia_ideologica.csv")
    print("Resultados salvos em 'resultados_aderencia_ideologica.csv'")


if __name__ == "__main__":
    # Configura o pandas para mostrar todas as colunas (útil para matrizes largas)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    main()

