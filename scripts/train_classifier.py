#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

LABEL_MAP = {"real": 0, "fake": 1}

# Lógica Complexa: A blocklist foi expandida para impedir o vazamento de 
# transformações globais (PCA), métricas derivadas que agrupam target e dados temporais.
LEAKAGE_COLUMNS = {
    "generation_method", 
    "anomaly_score", 
    "PC1", "PC2", 
    "Data", "date", "timestamp", "created_at"
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised deepfake classifier without leakage.")
    parser.add_argument("--input", default="data/processed/deepfake_dataset_cleaned.csv")
    parser.add_argument("--target", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fake-threshold", type=float, default=0.49)
    parser.add_argument("--max-cardinality", type=int, default=20, help="Maximum unique categories allowed per categorical feature.")
    parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=[
            "face_count",
            "lip_sync_score",
            "visual_artifacts_score",
            "compression_level",
            "lighting_inconsistency_score",
        ]
    )
    parser.add_argument("--min-signal-features", type=int, default=0)
    parser.add_argument("--out-model", default="data/processed/deepfake_classifier.joblib")
    parser.add_argument("--out-predictions", default="data/processed/deepfake_test_predictions.csv")
    parser.add_argument("--out-metrics", default="data/processed/deepfake_classifier_metrics.json")
    parser.add_argument("--out-confusion", default="assets/deepfake_confusion_matrix.png")
    parser.add_argument("--out-feature-importance", default="assets/deepfake_feature_importance.png")
    return parser.parse_args()

def normalize_target(y: pd.Series) -> pd.Series:
    if y.dtype == "object":
        normalized = y.astype(str).str.strip().str.lower().map(LABEL_MAP)
        if normalized.isna().any():
            unknown = sorted(y[normalized.isna()].astype(str).unique().tolist())
            raise ValueError(f"Target unsupported. Found: {unknown}")
        return normalized.astype(int)
    return y.astype(int)

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def build_pipeline(numeric_cols: list[str], categorical_cols: list[str], random_state: int) -> Pipeline:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        class_weight="balanced",
        max_depth=15, # Adicionado para forçar generalização e evitar overfitting profundo
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])

def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    y = normalize_target(df[args.target])
    X = df.drop(columns=[args.target]).copy()

    # 1. Expurgar IDs e colunas de blocklist (Data Leakage Control)
    non_predictive_cols = []
    if "media_id" in X.columns:
        non_predictive_cols.append("media_id")
    non_predictive_cols.extend(sorted(col for col in LEAKAGE_COLUMNS if col in X.columns))
    
    if non_predictive_cols:
        X = X.drop(columns=non_predictive_cols)
        print(f"Colunas de Blocklist/ID removidas para evitar leakage: {non_predictive_cols}")

    # 2. Filtro de Sinal (Quality Control)
    available_signal_cols = [c for c in args.signal_columns if c in X.columns]
    if args.min_signal_features > 0:
        signal_count = X[available_signal_cols].notna().sum(axis=1)
        informative_mask = signal_count >= args.min_signal_features
        X = X.loc[informative_mask].copy()
        y = y.loc[informative_mask].copy()

    # 3. Definição de Colunas e Trava de Cardinalidade (Data Leakage Control)
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    raw_categorical_cols = [c for c in X.columns if c not in numeric_cols]
    
    categorical_cols = []
    dropped_cat_cols = []
    
    # Lógica Complexa: Previne que o OneHotEncoder crie milhares de features a partir de
    # URLs, hashes ou descrições, o que faria o modelo "decorar" o dataset de treino.
    for col in raw_categorical_cols:
        unique_count = X[col].nunique()
        if unique_count <= args.max_cardinality:
            categorical_cols.append(col)
        else:
            dropped_cat_cols.append((col, unique_count))
            X = X.drop(columns=[col])
            
    if dropped_cat_cols:
        print(f"Atenção: Colunas ignoradas por alta cardinalidade (Leakage Risk): {dropped_cat_cols}")

    # 4. Train-Test Split (Isolamento de Variância)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # 5. Modelagem Segura
    pipeline = build_pipeline(numeric_cols, categorical_cols, args.random_state)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= args.fake_threshold).astype(int)

    # Coleta de Métricas e Exportação (Mantido igual ao original)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    for output_path in [args.out_model, args.out_predictions, args.out_metrics, args.out_confusion, args.out_feature_importance]:
        ensure_parent(output_path)

    joblib.dump(pipeline, args.out_model)

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    pred_df["prob_fake"] = y_proba
    pred_df.to_csv(args.out_predictions, index=False)

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Real", "Fake"], cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(args.out_confusion, dpi=200)
    plt.close()

    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_importances = importances.sort_values(ascending=False).head(15).sort_values()

    plt.figure(figsize=(9, 6))
    top_importances.plot(kind="barh", color="steelblue")
    plt.title("Top 15 Feature Importances - RandomForest")
    plt.tight_layout()
    plt.savefig(args.out_feature_importance, dpi=200)
    plt.close()

    print(f"Treinamento finalizado. Métricas limpas (Sem Leakage): AUC = {metrics['roc_auc']:.4f}")

if __name__ == "__main__":
    main()