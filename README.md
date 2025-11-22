**Projeto:** Análise de Votações do Senado (K-Means )

- **Descrição**: Projeto para coletar, filtrar e analisar votações nominais do Senado Federal (BR) usando K-Means para identificar padrões de adesão ideológica. Inclui coleta via API pública, limpeza/filtros inteligentes e pipeline de análise com PCA (redução de dimensionalidade) e validação de K.

**Requisitos**:
- Python 3.10+ (recomendado)
- `pip` (gerenciador de pacotes)
- Arquivo `requirements.txt` na raiz com as dependências do projeto

**Instalação (Windows - cmd.exe)**
- **1. Criar virtualenv**: 
```cmd
python -m venv .venv
```
- **2. Ativar virtualenv** (no cmd.exe):
```cmd
.venv\Scripts\activate
```
- **3. Atualizar pip (opcional, mas recomendado)**:
```cmd
python -m pip install --upgrade pip
```
- **4. Instalar dependências**:
```cmd
pip install -r requirements.txt
```
Se `requirements.txt` não existir ou estiver desatualizado, instale manualmente:
```cmd
pip install pandas numpy matplotlib scikit-learn requests
```


**Parâmetros importantes e personalização**
- `min_presenca_percent` em `filtrar_dataset_inteligente(...)` (arquivo `coletor_senado.py`): define o mínimo de presença exigido (por padrão 0.5 → 50%).
- `variancia_alvo` em `aplicar_pca(...)` (arquivo `analise_votacoes_kmeans.py`): porcentagem de variância a preservar no PCA (padrão 0.95 → 95%). Reduzir esse valor diminui mais a dimensão.
- `max_k` em `encontrar_k_otimo_melhorado(...)`: limite superior testado para K ao procurar K ótimo.

**Notas sobre normalização e PCA**
- Foi adicionada a função `aplicar_pca` que reduz a dimensionalidade antes do K-Means. Recomenda-se usar PCA quando o número de votações (colunas) .


**Estrutura do repositório (arquivos principais)**
- `coletor_senado.py`: coleta dados do Senado, aplica filtros inteligentes e salva CSV filtrado.
- `analise_votacoes_kmeans.py`: pipeline de análise — monta matriz senador×votação, aplica (ou não) normalização, PCA, encontra K ótimo, executa K-Means, calcula aderência ideológica e salva resultados.
- `dataset_votacoes_*.csv`: datasets já gerados (ex.: `dataset_votacoes_senado_2019_a_2023_FILTRADO.csv`).
- `requirements.txt`: lista de dependências (uso com `pip install -r requirements.txt`).
- `resultados_aderencia_ideologica.csv`: saída gerada pelo pipeline de análise.

