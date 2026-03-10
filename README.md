# Deepfake Detection Research

Projeto de pesquisa aplicada para detecção de deepfakes com foco em três frentes complementares:

- classificação supervisionada de conteúdo `Real` vs `Fake`;
- redução de dimensionalidade com PCA para análise exploratória e compactação de features;
- modelagem temporal do volume de deepfakes com `ARIMA` e `Prophet`;
- visualização analítica em dashboard `Streamlit`.

O repositório reúne notebook de pesquisa, scripts executáveis, artefatos processados, gráficos exportados e testes de smoke para validação básica da estrutura.

## Objetivos

- investigar sinais forenses presentes em mídia sintética;
- construir uma baseline reproduzível para classificação binária;
- analisar a separabilidade estatística das variáveis numéricas;
- gerar uma série temporal diária de ocorrência de deepfakes;
- comparar previsões e apoiar exploração visual em um painel interativo.

## Escopo do projeto

O fluxo implementado no repositório cobre:

1. limpeza e padronização do dataset;
2. treinamento de um classificador `RandomForestClassifier`;
3. exportação de previsões, métricas e gráficos de importância/confusão;
4. aplicação de `PCA` em variáveis numéricas já tratadas;
5. geração de série temporal diária de volume de deepfakes;
6. treinamento de modelo `Prophet` para previsão futura;
7. exploração interativa dos resultados em `Streamlit`.

O notebook também contém análise com `ARIMA`, embora o treinamento de ARIMA esteja hoje embutido no dashboard e não em um script dedicado no diretório `scripts/`.

## Estrutura do repositório

```text
.
├── assets/
│   ├── deepfake_confusion_matrix.png
│   ├── deepfake_feature_importance.png
│   ├── pairplot_separabilidade.png
│   ├── pca_scree_plot.png
│   ├── prophet_components.png
│   └── prophet_forecast.png
├── data/
│   ├── processed/
│   │   ├── deepfake_classifier.joblib
│   │   ├── deepfake_classifier_metrics.json
│   │   ├── deepfake_dataset_cleaned.csv
│   │   ├── deepfake_dataset_pca.csv
│   │   ├── deepfake_test_predictions.csv
│   │   ├── df_prophet.csv
│   │   ├── df_timeseries.csv
│   │   ├── pca_variance_summary.csv
│   │   └── prophet_forecast.csv
│   └── raw/
│       └── .gitkeep
├── notebook/
│   ├── app.py
│   ├── deepfake_forensics.py
│   └── deepfake_notebook.ipynb
├── scripts/
│   ├── generate_timeseries.py
│   ├── run_pca.py
│   ├── train_classifier.py
│   └── train_prophet.py
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

## Stack utilizada

- `Python`
- `pandas`, `numpy`
- `scikit-learn`
- `statsmodels`
- `prophet`
- `matplotlib`, `seaborn`
- `streamlit`, `plotly` no dashboard
- `unittest` para smoke tests

## Dataset e convenções

### Entrada principal

O pipeline supervisionado espera, por padrão, um arquivo em:

```bash
data/raw/deepfake_dataset.csv
```

Como `data/raw/` não versiona o dataset original, você precisa disponibilizar esse CSV localmente para reexecutar o treinamento do classificador a partir do zero.

### Dataset processado versionado

O repositório já inclui artefatos prontos em `data/processed/`, incluindo:

- `deepfake_dataset_cleaned.csv`: dataset limpo e normalizado;
- `deepfake_dataset_pca.csv`: projeção em componentes principais;
- `df_timeseries.csv`: série temporal diária de volume de deepfakes;
- `prophet_forecast.csv`: saída completa de previsão do Prophet;
- `deepfake_classifier.joblib`: pipeline serializado do classificador.

### Colunas relevantes

Os smoke tests validam que o dataset processado contenha pelo menos:

- `media_type`
- `content_category`
- `face_count`
- `audio_present`
- `lip_sync_score`
- `visual_artifacts_score`
- `compression_level`
- `lighting_inconsistency_score`
- `source_platform`
- `label`

### Rótulo

O classificador aceita `label` como:

- texto: `Real` / `Fake`;
- binário numérico: `0` / `1`.

Internamente, o mapeamento usado é:

- `real -> 0`
- `fake -> 1`

## Como preparar o ambiente

### 1. Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
.venv\Scripts\activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Dependências do dashboard

O arquivo `notebook/app.py` usa `streamlit` e `plotly`. Se esses pacotes não estiverem presentes no ambiente, instale também:

```bash
pip install streamlit plotly
```

## Pipeline de execução

A ordem recomendada para reproduzir os artefatos é a seguinte.

### 1. Treinar o classificador supervisionado

```bash
python scripts/train_classifier.py
```

Entrada padrão:

```bash
data/raw/deepfake_dataset.csv
```

Saídas padrão:

- `data/processed/deepfake_classifier.joblib`
- `data/processed/deepfake_test_predictions.csv`
- `data/processed/deepfake_classifier_metrics.json`
- `assets/deepfake_confusion_matrix.png`
- `assets/deepfake_feature_importance.png`

O script:

- valida a coluna alvo `label`;
- remove colunas não preditivas ou com risco de leakage, como `media_id` e `generation_method`;
- separa features numéricas e categóricas;
- aplica imputação em pipeline;
- usa `OneHotEncoder` para categorias;
- treina um `RandomForestClassifier` com `class_weight="balanced"`;
- gera probabilidades e converte para classe usando threshold configurável.

Parâmetros úteis:

```bash
python scripts/train_classifier.py \
  --input data/raw/deepfake_dataset.csv \
  --target label \
  --test-size 0.2 \
  --random-state 42 \
  --fake-threshold 0.49
```

Parâmetros adicionais relevantes:

- `--signal-columns`: define colunas usadas para medir sinal forense mínimo;
- `--min-signal-features`: filtra linhas com baixa informação;
- `--out-model`, `--out-predictions`, `--out-metrics`: personalizam saídas.

### 2. Rodar PCA

```bash
python scripts/run_pca.py
```

Entrada padrão:

```bash
data/processed/deepfake_dataset_cleaned.csv
```

Saídas padrão:

- `data/processed/deepfake_dataset_pca.csv`
- `data/processed/pca_variance_summary.csv`
- `assets/pca_scree_plot.png`

O script:

- seleciona apenas colunas numéricas;
- exclui `media_id` por ser identificador;
- calcula matriz de covariância e autovalores/autovetores;
- aplica `PCA` até atingir uma variância explicada acumulada alvo;
- preserva colunas contextuais, se existirem, como `label`, `media_type`, `content_category`, `audio_present` e `source_platform`.

Exemplo:

```bash
python scripts/run_pca.py --variance-threshold 0.95
```

### 3. Gerar série temporal

```bash
python scripts/generate_timeseries.py
```

Entrada padrão:

```bash
data/processed/deepfake_dataset_cleaned.csv
```

Saída padrão:

```bash
data/processed/df_timeseries.csv
```

O script funciona em dois modos:

- se encontrar uma coluna de data, usa as datas reais;
- se não encontrar, distribui amostras `Fake` ao longo de um intervalo simulado de dias.

Colunas de data reconhecidas automaticamente:

- `date`
- `data`
- `datetime`
- `timestamp`
- `created_at`
- `event_time`
- `event_date`

Exemplo:

```bash
python scripts/generate_timeseries.py \
  --start-date 2024-01-01 \
  --years 2 \
  --seed 42
```

### 4. Treinar previsão com Prophet

```bash
python scripts/train_prophet.py
```

Entrada padrão:

```bash
data/processed/df_timeseries.csv
```

Saídas padrão:

- `data/processed/df_prophet.csv`
- `data/processed/prophet_forecast.csv`
- `assets/prophet_forecast.png`
- `assets/prophet_components.png`

O script:

- espera colunas `Data` e `Volume_Deepfakes`;
- converte para o formato `ds`, `y` exigido pelo Prophet;
- treina um `Prophet()` padrão;
- projeta `30` dias no futuro por padrão.

Exemplo:

```bash
python scripts/train_prophet.py --periods 30
```

## Dashboard Streamlit

Para abrir o painel interativo:

```bash
streamlit run notebook/app.py
```

O dashboard consome os artefatos já gerados em `data/processed/` e oferece:

- filtro lateral de threshold de anomalia em metadados;
- visualização sobreposta de dados reais, previsão ARIMA e previsão Prophet;
- matriz de correlação interativa;
- narrativa visual dos sinais forenses mais fortes;
- carregamento do classificador serializado para uso no app.

Arquivos esperados pelo app:

- `data/processed/deepfake_dataset_cleaned.csv`
- `data/processed/deepfake_dataset_pca.csv`
- `data/processed/df_timeseries.csv`
- `data/processed/prophet_forecast.csv`
- `data/processed/deepfake_classifier.joblib`

## Notebook de pesquisa

O arquivo `notebook/deepfake_notebook.ipynb` concentra a exploração analítica do projeto. Pelas seções identificadas no notebook, ele cobre pelo menos:

- ingestão e profiling inicial;
- tratamento de nulos;
- auditoria e remoção de duplicatas;
- tratamento de outliers;
- normalização e exportação do dataset limpo;
- análise de separabilidade;
- modelagem ARIMA;
- avaliação comparativa `ARIMA vs Prophet`.

O arquivo `notebook/deepfake_forensics.py` é um rascunho de pipeline simplificado derivado do notebook, útil como referência, mas não é hoje o ponto principal de execução do projeto.

## Artefatos já presentes no repositório

### Classificação

Resultados atuais em `data/processed/deepfake_classifier_metrics.json`:

- `accuracy`: `0.8826`
- `precision`: `0.8098`
- `recall`: `1.0000`
- `f1`: `0.8949`
- `roc_auc`: `0.9698`
- `fake_threshold`: `0.49`

Distribuição registrada no split:

- treino: `529` reais e `527` fakes;
- teste: `132` reais e `132` fakes.

Observações do pipeline atual:

- `1320` linhas foram usadas em treino e avaliação;
- nenhuma linha foi descartada por `min_signal_features`, porque o valor salvo está em `0`;
- `media_id` e `generation_method` foram removidas antes do treinamento.

### PCA

Resumo atual de variância explicada em `data/processed/pca_variance_summary.csv`:

- `PC1`: `51.47%`
- `PC2`: `20.21%`
- `PC3`: `19.87%`
- `PC4`: `4.57%`
- variância acumulada em `4` componentes: `96.11%`

### Série temporal e forecasting

- `df_timeseries.csv` contém a série diária com colunas `Data` e `Volume_Deepfakes`;
- `prophet_forecast.csv` contém a previsão completa do Prophet com colunas como `ds`, `trend`, `yhat_lower`, `yhat_upper` e `yhat`.

## Testes e validação

### Rodar verificação local

```bash
python -m py_compile notebook/app.py scripts/*.py tests/test_smoke.py
python -m unittest tests.test_smoke -v
```

Os smoke tests validam:

- se os arquivos Python principais compilam;
- se os CSVs processados essenciais existem;
- se os CSVs têm as colunas mínimas esperadas.

### Integração contínua

O workflow em `.github/workflows/ci.yml` roda em `push` e `pull_request` e executa:

1. `python -m py_compile notebook/app.py scripts/*.py tests/test_smoke.py`
2. `python -m unittest tests.test_smoke -v`

A CI está configurada para `Python 3.13`.

## Arquivos principais

- `scripts/train_classifier.py`: treinamento supervisionado e exportação de métricas/plots;
- `scripts/run_pca.py`: redução de dimensionalidade e scree plot;
- `scripts/generate_timeseries.py`: geração de série temporal diária;
- `scripts/train_prophet.py`: previsão temporal com Prophet;
- `notebook/app.py`: dashboard Streamlit;
- `notebook/deepfake_notebook.ipynb`: notebook analítico principal;
- `tests/test_smoke.py`: validação mínima automatizada.

## Decisões de implementação

### Prevenção de leakage

O treinamento remove explicitamente:

- `media_id`
- `generation_method`

A intenção é evitar que identificadores ou pistas artificiais contaminem a aprendizagem do modelo.

### Pré-processamento do classificador

O pipeline do `scikit-learn` separa:

- numéricas: imputação por mediana;
- categóricas: imputação pela moda + one-hot encoding.

### Threshold de classificação

A classe `Fake` é atribuída com base em probabilidade e threshold configurável, hoje salvo como `0.49` no artefato versionado.

## Limitações atuais

- o dataset bruto não está versionado em `data/raw/`;
- o dashboard depende de artefatos pré-gerados em `data/processed/`;
- o comparativo `ARIMA vs Prophet` está mais presente no notebook/app do que em scripts independentes;
- `streamlit` e `plotly` são usados pelo app, mas podem precisar de instalação adicional dependendo do ambiente;
- os testes atuais são smoke tests, não testes funcionais profundos do comportamento estatístico dos modelos.

## Reprodutibilidade recomendada

Para recriar o fluxo completo em uma máquina limpa:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit plotly
python scripts/train_classifier.py
python scripts/run_pca.py
python scripts/generate_timeseries.py
python scripts/train_prophet.py
python -m unittest tests.test_smoke -v
streamlit run notebook/app.py
```

## Próximos passos naturais

- adicionar versionamento controlado ou instruções de aquisição do dataset bruto;
- transformar a etapa ARIMA em script dedicado em `scripts/`;
- expandir a cobertura de testes para métricas e contratos de saída;
- incluir avaliação comparativa formal entre modelos de classificação.
