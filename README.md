# Deepfake Detection Research

Pipeline de dados e modelagem para detecção de deepfakes, com:
- classificação supervisionada (RandomForest),
- redução de dimensionalidade (PCA),
- séries temporais (ARIMA + Prophet),
- dashboard em Streamlit para análise e teste de inferência.

## Estrutura

```text
deepfake-detection-research/
├── assets/                          # Gráficos exportados
├── data/
│   ├── processed/                   # Artefatos processados e métricas
│   └── raw/                         # Dados brutos locais (ignorado no Git)
├── notebook/
│   ├── app.py                       # Dashboard Streamlit
│   └── deepfake_notebook.ipynb      # EDA
├── scripts/
│   ├── generate_timeseries.py
│   ├── run_pca.py
│   ├── train_classifier.py
│   └── train_prophet.py
├── tests/
│   └── test_smoke.py                # Smoke tests de estrutura/artefatos
├── .github/workflows/ci.yml         # CI
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows CMD
pip install -r requirements.txt
```

## Execução do pipeline

```bash
python scripts/train_classifier.py
python scripts/run_pca.py
python scripts/generate_timeseries.py
python scripts/train_prophet.py
```

## Dashboard

```bash
streamlit run notebook/app.py
```

## Qualidade e CI

Smoke checks locais:

```bash
python -m py_compile notebook/app.py scripts/*.py tests/test_smoke.py
python -m unittest tests.test_smoke -v
```

O CI roda automaticamente em push/pull request e valida:
- compilação dos scripts e app,
- consistência mínima dos arquivos de dados processados.

## Notas de governança de dados

- `data/raw/` está ignorado por padrão para evitar versionar datasets sensíveis/pesados.
- os arquivos em `data/processed/` podem ser versionados para reprodutibilidade de resultados.
