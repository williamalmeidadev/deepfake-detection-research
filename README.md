# 🔍 Deepfake Detection Research

Pipeline analítica ponta a ponta para detecção e previsão de mídias sintéticas (Deepfakes). Este repositório contém a infraestrutura de extração, limpeza (Z-Score), redução de dimensionalidade (PCA) e modelagem preditiva de séries temporais (Prophet e ARIMA).

## 🏗️ Arquitetura do Projeto

A estrutura de diretórios foi desenhada para garantir reprodutibilidade absoluta, isolando dados brutos, scripts de automação e notebooks de análise exploratória.

> **Regra de Ouro (Data Governance):** O diretório `.venv` e os dados brutos pesados NUNCA sobem para este repositório. O rastreamento é bloqueado via `.gitignore`.

```text
deepfake-detection-research/
├── assets/                 # Gráficos exportados (Pairplots, Scree Plots, Forecasts)
├── data/
│   ├── processed/          # CSVs limpos e séries temporais geradas
│   └── raw/                # (Ignorado no Git) Datasets originais
├── notebook/
│   └── deepfake_notebook.ipynb  # EDA e modelagem ARIMA interativa
├── scripts/                # Scripts Python puros para execução em batch
│   ├── generate_timeseries.py
│   ├── run_pca.py
│   └── train_prophet.py
├── .gitignore              # Blindagem de arquivos locais
├── requirements.txt        # Contrato de dependências exatas
└── README.md               # Este manual
```

## 🚀 Setup e Execução (Clone and Play)
Para garantir que o código rode perfeitamente na sua máquina, siga a ordem estrita de comandos abaixo para construir a infraestrutura local.

### 1. Clonagem e Isolamento
Abra o seu terminal (Recomendado: Command Prompt ou PowerShell no Windows, Bash no Linux/Mac) e execute:
```bash
# Baixe o repositório
git clone [https://github.com/seu-usuario/deepfake-detection-research.git](https://github.com/seu-usuario/deepfake-detection-research.git)
cd deepfake-detection-research

# Construa o contêiner isolado do Python
python -m venv .venv
```

### 2. Ativação do Ambiente
Antes de rodar qualquer script, você **DEVE** ativar o ambiente virtual. Isso garante que você use as versões exatas das bibliotecas instaladas (ex: Pandas 3.0.1, Prophet 1.3.0) sem conflitar com outros projetos no seu sistema.

**Windows (Command Prompt):**
```bash
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux / MacOS:**
```bash
source .venv/bin/activate
```

### 3. Instalação de Dependências
Com o ambiente ativo (você verá `(.venv)` no início da linha do terminal), instale as bibliotecas listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

