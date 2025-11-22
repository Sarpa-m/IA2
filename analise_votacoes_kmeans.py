import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# A. COLETA E ESTRUTURAÇÃO DOS DADOS - VERSÃO EXPANDIDA
# ============================================================================

def carregar_dados_csv(caminho_arquivo: str) -> pd.DataFrame:
    """Carrega dados de um arquivo CSV real."""
    print(f"Carregando dados de {caminho_arquivo}...")
    try:
        df = pd.read_csv(caminho_arquivo)
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{caminho_arquivo}'")
        return pd.DataFrame() 
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        return pd.DataFrame()

def mapear_votos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte votos categóricos em numéricos.
    Sim: +1, Não: -1, Outros: 0
    """
    mapa_votos = {
        'Sim': 1,
        'Não': -1,
        'Abstenção': 0,
        'AP': 0,
        'Ausência': 0
    }
    
    # .fillna(0) garante que qualquer outro voto (Obstrução, etc.) vire Nulo
    df['voto_numerico'] = df['sigla_voto'].map(mapa_votos).fillna(0).astype(int)
    return df

def criar_matriz_senador_votacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma o DataFrame em matriz senador-votação.
    """
    print("Criando matriz senador-votação...")
    
    matriz = df.pivot_table(
        index='codigo_parlamentar',
        columns='codigo_votacao',
        values='voto_numerico',
        fill_value=0
    )
    
    # Preservar nomes dos parlamentares
    mapa_nomes = df.drop_duplicates('codigo_parlamentar').set_index(
        'codigo_parlamentar'
    )['nome_parlamentar']
    
    mapa_partidos = df.drop_duplicates('codigo_parlamentar').set_index(
        'codigo_parlamentar'
    )['partido_parlamentar']
    
    matriz.attrs['nomes'] = mapa_nomes
    matriz.attrs['partidos'] = mapa_partidos
    
    return matriz

def pre_processar(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de pré-processamento."""
    df = mapear_votos(df)
    matriz = criar_matriz_senador_votacao(df)
    return matriz

# ============================================================================
# B. NORMALIZAÇÃO E ANÁLISE DESCRITIVA
# ============================================================================

def analisar_variancia(matriz: pd.DataFrame) -> dict:
    """
    Analisa a variância dos dados para entender a estrutura.
    """
    print("\n" + "="*60)
    print("ANÁLISE DESCRITIVA DOS DADOS")
    print("="*60)
    
    stats = {
        'n_parlamentares': matriz.shape[0],
        'n_votacoes': matriz.shape[1],
        'sparsidade': (matriz == 0).sum().sum() / (matriz.shape[0] * matriz.shape[1]),
        'variancia_media': matriz.var(axis=0).mean(),
        'variancia_total': matriz.var().sum()
    }
    
    print(f"Número de parlamentares: {stats['n_parlamentares']}")
    print(f"Número de votações: {stats['n_votacoes']}")
    print(f"Sparsidade (% de zeros): {stats['sparsidade']*100:.2f}%")
    print(f"Variância média por votação: {stats['variancia_media']:.4f}")
    print(f"Variância total: {stats['variancia_total']:.4f}")
    
    return stats

def normalizar_dados(matriz: pd.DataFrame) -> np.ndarray:
    """
    Normaliza os dados usando StandardScaler.
    Crucial para K-means convergir melhor.
    """
    print("\nNormalizando dados...")
    scaler = StandardScaler()
    dados_normalizados = scaler.fit_transform(matriz)
    print(f"Dados normalizados: média={dados_normalizados.mean():.6f}, "
          f"std={dados_normalizados.std():.6f}")
    return dados_normalizados


def aplicar_pca(matriz_dados, variancia_alvo: float = 0.95):
    """
    Aplica PCA para reduzir o ruído e a dimensionalidade excessiva.
    Retorna (dados_reduzidos, pca_model).
    """
    print(f"\n[METODOLOGIA] Aplicando PCA (Preservando {variancia_alvo*100}% da variância)...")
    pca = PCA(n_components=variancia_alvo, random_state=42)
    dados_pca = pca.fit_transform(matriz_dados)

    # Informações úteis
    n_original = getattr(matriz_dados, 'shape', (None, None))[1]
    n_reduzido = dados_pca.shape[1]
    print(f"Dimensões originais: {n_original} votações")
    print(f"Dimensões reduzidas (PCA): {n_reduzido} componentes principais")
    return dados_pca, pca

# ============================================================================
# C. VALIDAÇÃO E OTIMIZAÇÃO 
# ============================================================================

def plotar_metricas_validacao(k_range: range, metricas: dict):
    """
    Plota múltiplas métricas de validação.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Método do Cotovelo
    axes[0, 0].plot(k_range, metricas['inertia'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Número de Clusters (K)', fontsize=11)
    axes[0, 0].set_ylabel('Inércia (WCSS)', fontsize=11)
    axes[0, 0].set_title('Método do Cotovelo', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Coeficiente de Silhueta
    axes[0, 1].plot(k_range, metricas['silhueta'], 'ro-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Número de Clusters (K)', fontsize=11)
    axes[0, 1].set_ylabel('Coeficiente de Silhueta', fontsize=11)
    axes[0, 1].set_title('Análise de Silhueta (maior é melhor)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Índice de Davies-Bouldin
    axes[1, 0].plot(k_range, metricas['davies_bouldin'], 'go-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Número de Clusters (K)', fontsize=11)
    axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=11)
    axes[1, 0].set_title('Davies-Bouldin (menor é melhor)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    
    # 4. Índice de Calinski-Harabasz
    axes[1, 1].plot(k_range, metricas['calinski_harabasz'], 'mo-', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Número de Clusters (K)', fontsize=11)
    axes[1, 1].set_ylabel('Calinski-Harabasz Index', fontsize=11)
    axes[1, 1].set_title('Calinski-Harabasz (maior é melhor)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle('Métricas de Validação de Clustering', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    plt.savefig("validacao_k_otimo_completa.png", dpi=300, bbox_inches='tight')
    print("Gráfico de validação salvo como 'validacao_k_otimo_completa.png'")
    plt.show()

def encontrar_k_otimo_melhorado(dados_normalizados: np.ndarray, max_k: int = 15) -> tuple:
    """
    Testa diferentes valores de K usando múltiplas métricas.
    Retorna K ótimo e um dicionário com todas as métricas.
    """
    print(f"\n{'='*60}")
    print("DETERMINANDO K ÓTIMO")
    print(f"{'='*60}")
    
    n_samples = dados_normalizados.shape[0]
    
    # Limitar max_k a um valor razoável
    max_k = min(max_k, n_samples // 2)
    
    if max_k < 2:
        print("Aviso: Dados insuficientes para clusterização")
        return 2, {}
    
    k_range = range(2, max_k + 1)
    metricas = {
        'inertia': [],
        'silhueta': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }
    
    print(f"Testando K de 2 até {max_k}...\n")
    print(f"{'K':<4} {'Inércia':<12} {'Silhueta':<12} {'Davies-B':<12} {'Calinski-H':<12}")
    print("-" * 52)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', 
                       random_state=42, n_init=20, max_iter=500)
        kmeans.fit(dados_normalizados)
        
        labels = kmeans.labels_
        
        # Calcular métricas
        inertia = kmeans.inertia_
        silhueta = silhouette_score(dados_normalizados, labels)
        davies_bouldin = davies_bouldin_score(dados_normalizados, labels)
        calinski_harabasz = calinski_harabasz_score(dados_normalizados, labels)
        
        metricas['inertia'].append(inertia)
        metricas['silhueta'].append(silhueta)
        metricas['davies_bouldin'].append(davies_bouldin)
        metricas['calinski_harabasz'].append(calinski_harabasz)
        
        print(f"{k:<4} {inertia:<12.2f} {silhueta:<12.4f} {davies_bouldin:<12.4f} {calinski_harabasz:<12.2f}")
    
    # Plotar gráficos
    plotar_metricas_validacao(k_range, metricas)
    
    # Escolher K baseado em múltiplas métricas (abordagem ensemble)
    silhueta_scores = np.array(metricas['silhueta'])
    k_silhueta = k_range[np.argmax(silhueta_scores)]
    
    davies_scores = np.array(metricas['davies_bouldin'])
    k_davies = k_range[np.argmin(davies_scores)]
    
    calinski_scores = np.array(metricas['calinski_harabasz'])
    k_calinski = k_range[np.argmax(calinski_scores)]
    
    # Votação: qual K aparece mais?
    votos = [k_silhueta, k_davies, k_calinski]
    k_otimo = max(set(votos), key=votos.count)
    
    print(f"\n{'='*60}")
    print("RECOMENDAÇÕES:")
    print(f"{'='*60}")
    print(f"K sugerido por Silhueta:       {k_silhueta}")
    print(f"K sugerido por Davies-Bouldin: {k_davies}")
    print(f"K sugerido por Calinski-H:    {k_calinski}")
    print(f"\n>>> K ÓTIMO (CONSENSO): {k_otimo} <<<")
    
    return k_otimo, metricas

# ============================================================================
# D. ALGORITMO DE CLASSIFICAÇÃO PARA ADERÊNCIA IDEOLÓGICA
# ============================================================================

def definir_voto_esperado(centroids: np.ndarray, tau: float = 0.1) -> np.ndarray:
    """
    Trinariza os centróides para definir o Voto Esperado.
    """
    votos_esperados = np.where(
        centroids > tau, 1, np.where(centroids < -tau, -1, 0)
    )
    return votos_esperados

def calcular_aderencia_ideologica(dados_normalizados: np.ndarray, 
                                  matriz_original: pd.DataFrame, 
                                  k_otimo: int) -> pd.DataFrame:
    """
    Executa K-Means final e calcula a Aderência Ideológica.
    """
    
    kmeans_final = KMeans(n_clusters=k_otimo, init='k-means++', 
                         random_state=42, n_init=30, max_iter=500)
    clusters = kmeans_final.fit_predict(dados_normalizados)
    
    centroids = kmeans_final.cluster_centers_
    votos_esperados_clusters = definir_voto_esperado(centroids, tau=0.1)
    
    print("Calculando Aderência Ideológica (AI) via Similaridade de Cosseno...\n")
    
    ai_scores = []
    
    nomes = matriz_original.attrs.get('nomes', {})
    partidos = matriz_original.attrs.get('partidos', {})
    
    for i in range(dados_normalizados.shape[0]):
        codigo_parlamentar = matriz_original.index[i]
        
        vetor_senador = dados_normalizados[i].reshape(1, -1)
        cluster_atribuido = clusters[i]
        vetor_esperado = votos_esperados_clusters[cluster_atribuido].reshape(1, -1)
        
        # Evitar divisão por zero
        if np.linalg.norm(vetor_senador) == 0 or np.linalg.norm(vetor_esperado) == 0:
            ai_score = 0.0
        else:
            ai_score = cosine_similarity(vetor_senador, vetor_esperado)[0][0]
        
        nome = nomes.get(codigo_parlamentar, f"Parlamentar_{codigo_parlamentar}")
        partido = partidos.get(codigo_parlamentar, "Desconhecido")
        
        ai_scores.append({
            'codigo_parlamentar': codigo_parlamentar,
            'nome_parlamentar': nome,
            'partido': partido,
            'cluster': cluster_atribuido,
            'aderencia_ideologica': ai_score
        })
    
    df_resultados = pd.DataFrame(ai_scores)
    
    # Estatísticas por cluster
    print("RESUMO POR CLUSTER:")
    print("-" * 80)
    for cluster_id in range(k_otimo):
        cluster_data = df_resultados[df_resultados['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data)} membros)")
        print(f"  Aderência média: {cluster_data['aderencia_ideologica'].mean():.4f}")
        print(f"  Partidos principais: {cluster_data['partido'].value_counts().head(3).to_dict()}")
    
    return df_resultados

# ============================================================================
# FUNÇÃO PRINCIPAL (MODIFICADA PARA O ORQUESTRADOR)
# ============================================================================

def executar_analise_pipeline(nome_arquivo_dados: str):
    """Executa o pipeline completo."""
    
    # --- FASE 1: CARREGAMENTO DE DADOS ---
    print("\n" + "="*60)
    print("FASE 1: CARREGAMENTO DE DADOS")
    print("="*60 + "\n")
    
    df_bruto = carregar_dados_csv(nome_arquivo_dados)
    
    if df_bruto.empty:
        print("Erro: Não foi possível carregar os dados.")
        print(f"Verifique se o arquivo '{nome_arquivo_dados}' está no mesmo diretório.")
        return
    
    print(f"Dados carregados: {df_bruto.shape[0]} registros de votação")
    
    # --- FASE 2: PRÉ-PROCESSAMENTO ---
    print("\n" + "="*60)
    print("FASE 2: PRÉ-PROCESSAMENTO")
    print("="*60 + "\n")
    
    matriz_votacoes = pre_processar(df_bruto)
    print(f"Matriz criada: {matriz_votacoes.shape}")
    
    # Análise Descritiva e Normalização
    stats = analisar_variancia(matriz_votacoes)
    dados_normalizados = normalizar_dados(matriz_votacoes)
    # Aplicar PCA para reduzir dimensionalidade e ruído antes do K-Means
    dados_processados, pca_model = aplicar_pca(dados_normalizados, variancia_alvo=0.95)
    # --- FASE 3: OTIMIZAÇÃO DO K ---
    print("\n" + "="*60)
    print("FASE 3: OTIMIZAÇÃO DO K")
    print("="*60)
    
    k_otimo, metricas = encontrar_k_otimo_melhorado(dados_processados, max_k=10)
    
    # --- FASE 4: DECISÃO MANUAL DO K ---
    print("\n" + "="*60)
    print("FASE 4: DECISÃO DO K FINAL")
    print("="*60 + "\n")
    
    k_sugerido = k_otimo
    k_final = k_sugerido # Valor padrão
    
    k_input_usuario = input(f">>> O K sugerido é {k_sugerido}. Pressione ENTER para aceitar ou digite um valor manual: ")
    
    if k_input_usuario.strip(): # Verifica se o usuário digitou algo
        try:
            k_manual = int(k_input_usuario)
            if k_manual >= 2: # K deve ser 2 ou mais
                k_final = k_manual
                print(f"--- Valor de K alterado manualmente para {k_final} ---")
            else:
                print(f"Valor '{k_manual}' é inválido (deve ser >= 2). Usando K sugerido: {k_sugerido}")
        except ValueError:
            print(f"Entrada '{k_input_usuario}' não é um número. Usando K sugerido: {k_sugerido}")
    else:
        print(f"--- K sugerido ({k_sugerido}) aceito ---")
    
    # --- FASE 5: CÁLCULO DA ADERÊNCIA IDEOLÓGICA ---
    print(f"\n{'='*60}")
    print(f"FASE 5: CLUSTERING FINAL COM K={k_final}")
    print(f"{'='*60}\n")
    
    df_resultados = calcular_aderencia_ideologica(dados_processados, 
                                                   matriz_votacoes, 
                                                   k_final) # Usa o K decidido
    
    # --- FASE 6: RESULTADOS FINAIS ---
    print("\n" + "="*60)
    print("FASE 6: RESULTADOS FINAIS")
    print("="*60 + "\n")
    
    print("Top 15 Parlamentares Mais Alinhados:")
    print(df_resultados.nlargest(15, 'aderencia_ideologica')[
        ['nome_parlamentar', 'partido', 'cluster', 'aderencia_ideologica']
    ].to_string(index=False))
    
    print("\n\nTop 15 Parlamentares Menos Alinhados:")
    print(df_resultados.nsmallest(15, 'aderencia_ideologica')[
        ['nome_parlamentar', 'partido', 'cluster', 'aderencia_ideologica']
    ].to_string(index=False))
    
    # Salvar resultados
    
    print("\n\nOrdenando resultados por cluster e salvando...")
    
    df_resultados_ordenado = df_resultados.sort_values(
        by=['cluster', 'aderencia_ideologica'], 
        ascending=[True, False]
    )
    
    # [MODIFICAÇÃO AQUI] Adiciona encoding para Excel
    df_resultados_ordenado.to_csv(
        "resultados_aderencia_ideologica.csv", 
        sep=";",
        index=False, 
        encoding='utf-8-sig'
    )
    
    print("Resultados salvos em 'resultados_aderencia_ideologica.csv'")


if __name__ == "__main__":
   
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    
    # --- Configuração Padrão (apenas se rodar este script direto) ---
    ANO_INICIO_PADRAO = 2023
    ANO_FIM_PADRAO = 2024
    NOME_ARQUIVOK_PADRAO = f"dataset_votacoes_senado_{ANO_INICIO_PADRAO}_a_{ANO_FIM_PADRAO}_FILTRADO.csv"
    # -----------------------------------------------------------------
    
    print("--- Executando 'analise_votacoes_kmeans.py' em modo standalone ---")
    executar_analise_pipeline(NOME_ARQUIVOK_PADRAO)