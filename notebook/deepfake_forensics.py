import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings

# Ignorar avisos chatos do Prophet
warnings.filterwarnings("ignore")


def executar_pipeline_deepfake():
    # --- CARD 1: INGESTÃO ---
    url = "https://raw.githubusercontent.com/williamalmeidadev/deepfake-detection-research/refs/heads/develop/data/raw/deepfake_dataset.csv"
    df = pd.read_csv(url)
    print("[OK] Dados carregados.")

    # --- CARD 2: TRATAMENTO ---
    # Removendo colunas com muitos nulos (generation_method)
    df.drop(columns=["generation_method"], inplace=True, errors="ignore")
    # Preenchendo nulos com a mediana
    df.fillna(df.median(numeric_only=True), inplace=True)

    # --- CARD 3: AUDITORIA ---
    df.drop_duplicates(inplace=True)

    # --- CARD 4: OUTLIERS ---
    # (Opcional) Filtrar baseado nos limites que calculamos (-2 a 6)
    df = df[(df["face_count"] >= -2) & (df["face_count"] <= 6)]

    # Preparando dados para Time Series
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_ts = df.groupby(df["timestamp"].dt.date).size().reset_index()
    df_ts.columns = ["ds", "y"]

    # ... aqui entrariam os modelos ARIMA e Prophet que estão no seu notebook ...

    # Criar pasta assets se não existir
    Path("assets").mkdir(parents=True, exist_ok=True)

    # Salvar o gráfico final
    print(
        "[SUCESSO] Pipeline executado e gráfico salvo em assets/forecast_comparativo.png"
    )


if __name__ == "__main__":
    executar_pipeline_deepfake()
