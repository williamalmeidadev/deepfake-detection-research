from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "deepfake_dataset_cleaned.csv"
PCA_DATA_PATH = PROCESSED_DIR / "deepfake_dataset_pca.csv"
TIMESERIES_PATH = PROCESSED_DIR / "df_timeseries.csv"
PROPHET_FORECAST_PATH = PROCESSED_DIR / "prophet_forecast.csv"
CLASSIFIER_PATH = PROCESSED_DIR / "deepfake_classifier.joblib"

st.set_page_config(
    page_title="Deepfake Forensics", layout="wide", initial_sidebar_state="expanded"
)

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

st.title("Sistema de Detecção de Deepfake Forensics")
st.markdown("Painel analítico com métricas e previsões do projeto de deepfake.")


@st.cache_data(show_spinner=False)
def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
    df_pca = pd.read_csv(PCA_DATA_PATH)
    df_timeseries = pd.read_csv(TIMESERIES_PATH)
    df_prophet_forecast = pd.read_csv(PROPHET_FORECAST_PATH)
    return df_cleaned, df_pca, df_timeseries, df_prophet_forecast


@st.cache_resource(show_spinner=False)
def load_classifier():
    return joblib.load(CLASSIFIER_PATH)


@st.cache_data(show_spinner=False)
def compute_arima_forecast(
    y_values: tuple[float, ...],
    forecast_start_date: str,
    forecast_steps: int,
) -> pd.DataFrame:
    import pmdarima as pm
    import pandas as pd
    import numpy as np

    if len(y_values) < 5:
        raise ValueError("Série temporal insuficiente para ajuste ARIMA.")
    if forecast_steps <= 0:
        raise ValueError("forecast_steps deve ser maior que zero.")

    y_series = pd.Series(y_values, dtype="float64")
    
    # Lógica Complexa: Utilização do método stepwise do auto_arima para varrer o 
    # espaço de hiperparâmetros (p,d,q) minimizando o critério de informação (AIC).
    # O modelo anterior era estático e falhava ao tentar forçar convergência linear.
    auto_model = pm.auto_arima(
        y_series,
        seasonal=True,
        m=7, # Informa ao modelo que existe um ciclo de 7 dias (semanal)
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )

    arima_values = auto_model.predict(n_periods=forecast_steps)
    
    # Pós-processamento crítico: Contagem de deepfakes não pode ser negativa.
    arima_values = np.clip(arima_values, a_min=0, a_max=None)
    
    arima_dates = pd.date_range(
        start=pd.to_datetime(forecast_start_date), periods=forecast_steps, freq="D"
    )
    return pd.DataFrame({"ds": arima_dates, "yhat": arima_values})


def render_analytics_tab(
    df_cleaned: pd.DataFrame,
    df_pca: pd.DataFrame,
    df_timeseries: pd.DataFrame,
    df_prophet_forecast: pd.DataFrame,
) -> pd.DataFrame:
    st.header("Desempenho do Modelo (Fase 3)")
    st.markdown("Visualizações interativas para análise forense (hover e zoom).")

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

        forecast_steps = max(len(prophet_future), 180)

        try:
            df_arima_plot = compute_arima_forecast(
                y_values=tuple(df_timeseries_plot["y"].astype(float).tolist()),
                forecast_start_date=(
                    max_real_date + pd.Timedelta(days=1)
                ).strftime("%Y-%m-%d"),
                forecast_steps=int(forecast_steps),
            )
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

    st.subheader("Storytelling Forense: Onde os sinais estão mais fortes")

    label_col = None
    if "label" in df_cleaned_filtered.columns:
        label_col = "label"

    if label_col:
        df_story = df_cleaned_filtered.copy()
        
        # Lógica Complexa: Transforma a coluna em string, remove o ".0" do final 
        # via regex caso exista, passa para maiúsculo e mapeia de forma robusta.
        df_story["class_label"] = (
            df_story[label_col]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
            .str.upper()
            .map({
                "0": "REAL",
                "1": "FAKE",
                "REAL": "REAL",
                "FAKE": "FAKE"
            })
            .fillna("N/A")
        )
    else:
        df_story = df_cleaned_filtered.copy()
        df_story["class_label"] = "N/A"

    col_artifacts, col_platform = st.columns(2)

    with col_artifacts:
        if {"visual_artifacts_score", "class_label"}.issubset(df_story.columns):
            fig_artifacts = px.box(
                df_story,
                x="class_label",
                y="visual_artifacts_score",
                color="class_label",
                points="all",
                title="Distribuição de Artefatos Visuais por Classe",
                labels={
                    "class_label": "Classe",
                    "visual_artifacts_score": "Score de Artefatos",
                },
                color_discrete_map={"REAL": "#1f77b4", "FAKE": "#d62728"},
            )
            fig_artifacts.update_layout(showlegend=False)
            st.plotly_chart(fig_artifacts, use_container_width=True)
        else:
            st.info("Dados insuficientes para distribuição de artefatos visuais.")

    with col_platform:
        if {"source_platform", "class_label"}.issubset(df_story.columns):
            df_platform_mix = (
                df_story.groupby(["source_platform", "class_label"], dropna=False)
                .size()
                .reset_index(name="count")
            )
            fig_platform_mix = px.bar(
                df_platform_mix,
                x="source_platform",
                y="count",
                color="class_label",
                barmode="stack",
                title="Volume por Rede Social e Classe",
                labels={
                    "source_platform": "Rede Social",
                    "count": "Volume",
                    "class_label": "Classe",
                },
                color_discrete_map={"REAL": "#1f77b4", "FAKE": "#d62728"},
            )
            st.plotly_chart(fig_platform_mix, use_container_width=True)
        else:
            st.info("Dados insuficientes para distribuição por rede social.")

    if {"source_platform", "content_category", "class_label"}.issubset(df_story.columns):
        df_fake_rate = (
            df_story.assign(is_fake=(df_story["class_label"] == "FAKE").astype(float))
            .groupby(["source_platform", "content_category"], dropna=False)["is_fake"]
            .mean()
            .reset_index(name="fake_rate")
        )
        pivot_fake_rate = df_fake_rate.pivot(
            index="source_platform", columns="content_category", values="fake_rate"
        )
        fig_fake_rate = px.imshow(
            pivot_fake_rate,
            color_continuous_scale="Reds",
            zmin=0,
            zmax=1,
            text_auto=".0%",
            aspect="auto",
            title="Heatmap: Taxa de Fake por Rede Social x Categoria",
            labels={"x": "Categoria", "y": "Rede Social", "color": "Taxa de Fake"},
        )
        st.plotly_chart(fig_fake_rate, use_container_width=True)

        if not df_fake_rate.empty:
            top_cell = df_fake_rate.sort_values("fake_rate", ascending=False).iloc[0]
            st.caption(
                "Insight: maior concentração de suspeitas em "
                f"{top_cell['source_platform']} / {top_cell['content_category']} "
                f"({top_cell['fake_rate']:.1%} de fake)."
            )
    else:
        st.info("Dados insuficientes para heatmap de taxa de fake por plataforma/categoria.")

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
    return df_cleaned_filtered


def render_prediction_tab(df_cleaned: pd.DataFrame) -> None:
    st.header("Teste do Modelo de Classificação")
    st.markdown(
        "Preencha os campos abaixo para simular uma mídia e visualizar a predição do classificador (Real/Fake)."
    )

    if not CLASSIFIER_PATH.exists():
        st.warning(
            f"Modelo não encontrado em `{CLASSIFIER_PATH}`. Rode `scripts/train_classifier.py` para gerar o artefato."
        )
        return

    try:
        model = load_classifier()
    except Exception as exc:
        st.error(f"Falha ao carregar o modelo serializado: {exc}")
        return

    feature_cols = list(getattr(model, "feature_names_in_", []))
    if not feature_cols:
        feature_cols = [
            c
            for c in df_cleaned.columns
            if c not in {"label", "media_id", "generation_method", "anomaly_score"}
        ]

    form_defaults = {}
    for col in feature_cols:
        if col not in df_cleaned.columns:
            continue
        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
            form_defaults[col] = float(pd.to_numeric(df_cleaned[col], errors="coerce").median())
        else:
            modes = df_cleaned[col].dropna().astype(str).mode()
            form_defaults[col] = modes.iloc[0] if not modes.empty else ""

    with st.form("single_prediction_form"):
        st.subheader("Entrada Manual")
        inputs = {}

        for col in feature_cols:
            if col not in df_cleaned.columns:
                continue

            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                col_series = pd.to_numeric(df_cleaned[col], errors="coerce").dropna()
                min_val = float(col_series.min()) if not col_series.empty else -5.0
                max_val = float(col_series.max()) if not col_series.empty else 5.0
                if min_val == max_val:
                    min_val -= 1.0
                    max_val += 1.0

                inputs[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=float(form_defaults.get(col, 0.0)),
                )
            else:
                options = df_cleaned[col].dropna().astype(str).unique().tolist()
                options = sorted(options)
                default_value = str(form_defaults.get(col, options[0] if options else ""))
                if default_value and default_value not in options:
                    options = [default_value] + options

                inputs[col] = st.selectbox(
                    f"{col}",
                    options=options if options else [""],
                    index=(options.index(default_value) if default_value in options else 0),
                )

        submitted = st.form_submit_button("Testar predição")

    if not submitted:
        return

    input_df = pd.DataFrame([inputs])

    missing_cols = [c for c in feature_cols if c not in input_df.columns]
    for missing in missing_cols:
        input_df[missing] = pd.NA

    input_df = input_df[feature_cols]

    try:
        pred_class = int(model.predict(input_df)[0])
        pred_label = "FAKE" if pred_class == 1 else "REAL"

        if hasattr(model, "predict_proba"):
            prob_fake = float(model.predict_proba(input_df)[0][1])
        else:
            prob_fake = float("nan")

        col1, col2 = st.columns(2)
        col1.metric("Classe prevista", pred_label)
        if pd.notna(prob_fake):
            col2.metric("Probabilidade de FAKE", f"{prob_fake:.2%}")
        else:
            col2.metric("Probabilidade de FAKE", "N/A")

        if pred_label == "FAKE":
            st.error("Resultado: a amostra foi classificada como suspeita (FAKE).")
        else:
            st.success("Resultado: a amostra foi classificada como autêntica (REAL).")

        st.caption("Payload enviado ao modelo:")
        st.dataframe(input_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Falha ao executar a predição: {exc}")


try:
    df_cleaned, df_pca, df_timeseries, df_prophet_forecast = load_processed_data()
except FileNotFoundError as exc:
    st.error(f"Arquivo de dados não encontrado: {exc}")
except Exception as exc:
    st.error(f"Falha ao carregar os dados processados: {exc}")
else:
    tab_analytics, tab_test = st.tabs(["Painel Analítico", "Teste do Modelo"])

    with tab_analytics:
        render_analytics_tab(df_cleaned, df_pca, df_timeseries, df_prophet_forecast)

    with tab_test:
        render_prediction_tab(df_cleaned)
