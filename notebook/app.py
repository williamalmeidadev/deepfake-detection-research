import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from pathlib import Path

# Importe aqui suas bibliotecas do notebook (MTCNN, PyTorch/Tensorflow conforme seu modelo)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "deepfake_dataset_cleaned.csv"
PCA_DATA_PATH = PROCESSED_DIR / "deepfake_dataset_pca.csv"
TIMESERIES_PATH = PROCESSED_DIR / "df_timeseries.csv"
PROPHET_FORECAST_PATH = PROCESSED_DIR / "prophet_forecast.csv"

# 1. Configuração da Página (Critério: Layout Wide)
st.set_page_config(
    page_title="Deepfake Forensics", layout="wide", initial_sidebar_state="expanded"
)

# 2. Barra Lateral (Critério: st.sidebar)
with st.sidebar:
    st.title("🛡️ Painel de Controle")
    st.markdown("---")
    st.markdown("### Parâmetros do Modelo")

    anomaly_threshold_pct = st.slider(
        "Threshold de Anomalia dos Metadados (%)",
        min_value=50,
        max_value=100,
        value=85,
    )
    temporal_model_view = st.selectbox(
        "Visualizar previsão temporal",
        options=["Ambos", "Prophet", "ARIMA"],
        index=0,
    )

    st.info(
        "Este painel separa os filtros do conteúdo principal para evitar sobrecarga cognitiva."
    )

# 3. Título e Área Principal (Critério: st.title e st.markdown)
st.title("Sistema de Detecção de Deepfake Forensics")
st.markdown(
    "Analise a autenticidade de mídias digitais utilizando modelos de Deep Learning."
)

# 4. Organização por Guias (Critério: Dividido por guias/Tabs)
tab_upload, tab_analise, tab_metricas = st.tabs(
    ["📤 Upload", "🔍 Análise Forense", "📊 Métricas do Modelo"]
)

with tab_upload:
    st.header("Envio de Arquivo")
    uploaded_file = st.file_uploader(
        "Escolha uma imagem para análise...", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem Carregada", use_container_width=True)
        st.success("Arquivo carregado com sucesso! Vá para a aba 'Análise Forense'.")

with tab_analise:
    st.header("Resultado da Inferência")
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Detecção de Face")
            # Simulando o processo do seu notebook (MTCNN)
            st.info("Detectando faces e extraindo frames...")
            st.image(image, width=300)  # Aqui entraria o crop da face

        with col2:
            st.subheader("Classificação")
            # Aqui entraria a chamada do seu modelo .pth ou .h5
            resultado = "REAL"  # Exemplo de saída
            probabilidade = 0.98

            if resultado == "REAL":
                st.success(f"Resultado: {resultado} ({probabilidade*100:.2f}%)")
            else:
                st.error(f"Resultado: FAKE ({probabilidade*100:.2f}%)")
    else:
        st.warning("Por favor, faça o upload de uma imagem na aba anterior.")

with tab_metricas:
    st.header("Desempenho do Modelo (Fase 3)")
    st.markdown("Visualizações interativas para análise forense (hover e zoom).")

    try:
        df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
        df_pca = pd.read_csv(PCA_DATA_PATH)
        df_timeseries = pd.read_csv(TIMESERIES_PATH)
        df_prophet_forecast = pd.read_csv(PROPHET_FORECAST_PATH)
    except FileNotFoundError as exc:
        st.error(f"Arquivo de dados não encontrado: {exc}")
    except Exception as exc:
        st.error(f"Falha ao carregar os dados processados: {exc}")
    else:
        anomaly_cols = [
            col
            for col in [
                "lip_sync_score",
                "visual_artifacts_score",
                "lighting_inconsistency_score",
            ]
            if col in df_cleaned.columns
        ]
        df_cleaned_filtered = df_cleaned.copy()
        if anomaly_cols:
            df_cleaned_filtered["anomaly_score"] = (
                df_cleaned_filtered[anomaly_cols].abs().mean(axis=1)
            )
            anomaly_cutoff = df_cleaned_filtered["anomaly_score"].quantile(
                anomaly_threshold_pct / 100
            )
            df_cleaned_filtered = df_cleaned_filtered[
                df_cleaned_filtered["anomaly_score"] >= anomaly_cutoff
            ].copy()
        else:
            anomaly_cutoff = None

        st.caption(
            f"Filtro ativo: top {100 - anomaly_threshold_pct}% de anomalia | "
            f"Registros filtrados: {len(df_cleaned_filtered)}/{len(df_cleaned)}"
        )
        if anomaly_cutoff is not None:
            st.caption(f"Corte de anomalia aplicado: {anomaly_cutoff:.4f}")

        st.subheader("Série Temporal: Real vs Previsões (ARIMA e Prophet)")
        required_ts = {"Data", "Volume_Deepfakes"}
        required_pf = {"ds", "yhat"}

        if not required_ts.issubset(df_timeseries.columns):
            st.warning(
                "Colunas ausentes em df_timeseries.csv (esperado: Data, Volume_Deepfakes)."
            )
        elif not required_pf.issubset(df_prophet_forecast.columns):
            st.warning("Colunas ausentes em prophet_forecast.csv (esperado: ds, yhat).")
        else:
            df_timeseries_plot = (
                df_timeseries[["Data", "Volume_Deepfakes"]]
                .rename(columns={"Data": "ds", "Volume_Deepfakes": "y"})
                .copy()
            )
            df_timeseries_plot["ds"] = pd.to_datetime(df_timeseries_plot["ds"])
            df_timeseries_plot["y"] = pd.to_numeric(df_timeseries_plot["y"], errors="coerce")
            df_timeseries_plot = df_timeseries_plot.dropna(subset=["ds", "y"]).sort_values("ds")

            df_prophet_plot = df_prophet_forecast[["ds", "yhat"]].copy()
            df_prophet_plot["ds"] = pd.to_datetime(df_prophet_plot["ds"])
            df_prophet_plot["yhat"] = pd.to_numeric(df_prophet_plot["yhat"], errors="coerce")
            df_prophet_plot = df_prophet_plot.dropna(subset=["ds", "yhat"]).sort_values("ds")

            max_real_date = df_timeseries_plot["ds"].max()
            prophet_future = df_prophet_plot[df_prophet_plot["ds"] > max_real_date].copy()

            prophet_date_filter = None
            if temporal_model_view in ["Ambos", "Prophet"] and not prophet_future.empty:
                prophet_min_date = prophet_future["ds"].min().date()
                prophet_max_date = prophet_future["ds"].max().date()
                prophet_date_filter = st.sidebar.date_input(
                    "Data inicial da previsão Prophet",
                    value=prophet_min_date,
                    min_value=prophet_min_date,
                    max_value=prophet_max_date,
                )
                prophet_future = prophet_future[
                    prophet_future["ds"] >= pd.to_datetime(prophet_date_filter)
                ].copy()

            forecast_steps = max(len(prophet_future), 30)

            try:
                from statsmodels.tsa.arima.model import ARIMA

                arima_model = ARIMA(df_timeseries_plot["y"], order=(1, 1, 1))
                arima_fit = arima_model.fit()
                arima_values = arima_fit.forecast(steps=forecast_steps)
                arima_dates = pd.date_range(
                    start=max_real_date + pd.Timedelta(days=1), periods=forecast_steps, freq="D"
                )
                df_arima_plot = pd.DataFrame({"ds": arima_dates, "yhat": arima_values})
            except Exception as exc:
                st.error(f"Falha ao gerar previsão ARIMA: {exc}")
                df_arima_plot = pd.DataFrame(columns=["ds", "yhat"])

            fig_forecast = go.Figure()
            fig_forecast.add_trace(
                go.Scatter(
                    x=df_timeseries_plot["ds"],
                    y=df_timeseries_plot["y"],
                    mode="lines",
                    name="Dados Reais",
                )
            )
            if temporal_model_view in ["Ambos", "ARIMA"]:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=df_arima_plot["ds"],
                        y=df_arima_plot["yhat"],
                        mode="lines",
                        name="Previsão ARIMA",
                    )
                )
            if temporal_model_view in ["Ambos", "Prophet"]:
                fig_forecast.add_trace(
                    go.Scatter(
                        x=prophet_future["ds"],
                        y=prophet_future["yhat"],
                        mode="lines",
                        name="Previsão Prophet",
                    )
                )
            fig_forecast.update_layout(
                title="Sobreposição de Série Temporal: Real vs Modelos de Previsão",
                xaxis_title="Data",
                yaxis_title="Volume",
                legend_title="Séries",
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            if prophet_date_filter:
                st.caption(
                    f"Filtro Prophet aplicado a partir de: {prophet_date_filter.strftime('%Y-%m-%d')}"
                )

        st.subheader("Matriz de Correlação Interativa")
        numeric_cols = df_cleaned_filtered.select_dtypes(include=["number"]).columns.tolist()
        if "media_id" in numeric_cols:
            numeric_cols.remove("media_id")

        if not numeric_cols:
            st.warning("Não há colunas numéricas suficientes para gerar a correlação.")
        else:
            corr = df_cleaned_filtered[numeric_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                aspect="auto",
                title="Correlação entre Features Numéricas",
            )
            fig_corr.update_layout(coloraxis_colorbar_title="Correlação")
            st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Dispersão do PCA (PC1 x PC2)")
        if "PC1" not in df_pca.columns or "PC2" not in df_pca.columns:
            st.warning("Colunas PCA ausentes no dataset (esperado: PC1 e PC2).")
        else:
            df_plot = df_pca.copy()
            for col in [
                "label",
                "media_type",
                "content_category",
                "audio_present",
                "source_platform",
            ]:
                if col in df_plot.columns and col in df_cleaned_filtered.columns:
                    allowed_values = (
                        df_cleaned_filtered[col].astype(str).dropna().unique().tolist()
                    )
                    df_plot = df_plot[df_plot[col].astype(str).isin(allowed_values)]

            color_col = None

            if "label" in df_plot.columns:
                label_map = {
                    0: "REAL",
                    1: "FAKE",
                    "0": "REAL",
                    "1": "FAKE",
                    "real": "REAL",
                    "fake": "FAKE",
                    "REAL": "REAL",
                    "FAKE": "FAKE",
                }
                df_plot["Target_Real_Fake"] = (
                    df_plot["label"].astype(str).map(label_map).fillna(df_plot["label"].astype(str))
                )
                color_col = "Target_Real_Fake"

            hover_cols = [
                col
                for col in ["media_type", "content_category", "audio_present", "source_platform", "label"]
                if col in df_plot.columns
            ]

            if color_col:
                fig_pca = px.scatter(
                    df_plot,
                    x="PC1",
                    y="PC2",
                    color=color_col,
                    hover_data=hover_cols,
                    title="PCA - Separação entre classes",
                )
            else:
                fig_pca = px.scatter(
                    df_plot,
                    x="PC1",
                    y="PC2",
                    hover_data=hover_cols,
                    title="PCA - Distribuição dos pontos",
                )

            fig_pca.update_traces(marker={"size": 9, "opacity": 0.8})
            st.plotly_chart(fig_pca, use_container_width=True)

        st.subheader("Amostra dos Dados Filtrados")
        st.dataframe(df_cleaned_filtered.head(100), use_container_width=True)
