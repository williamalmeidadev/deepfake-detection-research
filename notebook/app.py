import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path

# Importe aqui suas bibliotecas do notebook (MTCNN, PyTorch/Tensorflow conforme seu modelo)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "deepfake_dataset_cleaned.csv"
PCA_DATA_PATH = PROCESSED_DIR / "deepfake_dataset_pca.csv"

# 1. Configuração da Página (Critério: Layout Wide)
st.set_page_config(
    page_title="Deepfake Forensics", layout="wide", initial_sidebar_state="expanded"
)

# 2. Barra Lateral (Critério: st.sidebar)
with st.sidebar:
    st.title("🛡️ Painel de Controle")
    st.markdown("---")
    st.markdown("### Parâmetros do Modelo")

    # Exemplo de ajuste de sensibilidade baseado no seu notebook
    confidence_threshold = st.slider("Limiar de Confiança", 0.0, 1.0, 0.5)

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
    except FileNotFoundError as exc:
        st.error(f"Arquivo de dados não encontrado: {exc}")
    except Exception as exc:
        st.error(f"Falha ao carregar os dados processados: {exc}")
    else:
        st.subheader("Matriz de Correlação Interativa")
        numeric_cols = df_cleaned.select_dtypes(include=["number"]).columns.tolist()
        if "media_id" in numeric_cols:
            numeric_cols.remove("media_id")

        if not numeric_cols:
            st.warning("Não há colunas numéricas suficientes para gerar a correlação.")
        else:
            corr = df_cleaned[numeric_cols].corr()
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
