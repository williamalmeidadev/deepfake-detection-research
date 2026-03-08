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
LEAKAGE_COLUMNS = {"generation_method"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised deepfake classifier (Real vs Fake)."
    )
    parser.add_argument(
        "--input",
        default="data/raw/deepfake_dataset.csv",
        help="Input CSV dataset path.",
    )
    parser.add_argument(
        "--target",
        default="label",
        help="Target column name.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size (0-1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split and model.",
    )
    parser.add_argument(
        "--fake-threshold",
        type=float,
        default=0.49,
        help="Probability threshold to classify a sample as Fake (class 1).",
    )
    parser.add_argument(
        "--signal-columns",
        nargs="+",
        default=[
            "face_count",
            "lip_sync_score",
            "visual_artifacts_score",
            "compression_level",
            "lighting_inconsistency_score",
        ],
        help="Columns used to measure if a row has enough forensic signal to be trainable.",
    )
    parser.add_argument(
        "--min-signal-features",
        type=int,
        default=0,
        help="Minimum number of non-null signal columns required to keep a sample (0 disables filtering).",
    )
    parser.add_argument(
        "--out-model",
        default="data/processed/deepfake_classifier.joblib",
        help="Output path for serialized model pipeline.",
    )
    parser.add_argument(
        "--out-predictions",
        default="data/processed/deepfake_test_predictions.csv",
        help="Output CSV with y_true, y_pred and probabilities.",
    )
    parser.add_argument(
        "--out-metrics",
        default="data/processed/deepfake_classifier_metrics.json",
        help="Output JSON with evaluation metrics.",
    )
    parser.add_argument(
        "--out-confusion",
        default="assets/deepfake_confusion_matrix.png",
        help="Output image for confusion matrix.",
    )
    parser.add_argument(
        "--out-feature-importance",
        default="assets/deepfake_feature_importance.png",
        help="Output image for top feature importances.",
    )
    return parser.parse_args()


def normalize_target(y: pd.Series) -> pd.Series:
    if y.dtype == "object":
        normalized = y.astype(str).str.strip().str.lower().map(LABEL_MAP)
        if normalized.isna().any():
            unknown = sorted(y[normalized.isna()].astype(str).unique().tolist())
            raise ValueError(
                "Target contains unsupported labels. Expected Real/Fake variants; "
                f"found: {unknown}"
            )
        return normalized.astype(int)

    unique_vals = set(pd.Series(y).dropna().unique().tolist())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Numeric target must be binary 0/1. Found values: {sorted(unique_vals)}")
    return y.astype(int)


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str], random_state: int) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
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
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])


def extract_feature_names(preprocess: ColumnTransformer) -> np.ndarray:
    return preprocess.get_feature_names_out()


def main() -> None:
    args = parse_args()

    if not (0 < args.test_size < 1):
        raise ValueError("--test-size must be in (0, 1).")
    if not (0 <= args.fake_threshold <= 1):
        raise ValueError("--fake-threshold must be in [0, 1].")

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")

    y = normalize_target(df[args.target])
    X = df.drop(columns=[args.target]).copy()

    if args.min_signal_features < 0:
        raise ValueError("--min-signal-features must be >= 0.")

    # Prevent identifier/target leakage into model learning.
    non_predictive_cols = []
    if "media_id" in X.columns:
        non_predictive_cols.append("media_id")
    non_predictive_cols.extend(sorted(col for col in LEAKAGE_COLUMNS if col in X.columns))
    if non_predictive_cols:
        X = X.drop(columns=non_predictive_cols)

    available_signal_cols = [c for c in args.signal_columns if c in X.columns]
    if not available_signal_cols:
        raise ValueError(
            "None of --signal-columns were found in dataset columns. "
            f"Provided: {args.signal_columns}"
        )

    if args.min_signal_features == 0:
        dropped_count = 0
    else:
        signal_count = X[available_signal_cols].notna().sum(axis=1)
        informative_mask = signal_count >= args.min_signal_features
        dropped_count = int((~informative_mask).sum())
        if informative_mask.sum() == 0:
            raise ValueError(
                "No rows satisfy the minimum signal threshold. "
                "Lower --min-signal-features or provide richer input data."
            )

        X = X.loc[informative_mask].copy()
        y = y.loc[informative_mask].copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No features available for training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols, args.random_state)
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= args.fake_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "fake_threshold": float(args.fake_threshold),
        "class_distribution": {
            "train_real": int((y_train == 0).sum()),
            "train_fake": int((y_train == 1).sum()),
            "test_real": int((y_test == 0).sum()),
            "test_fake": int((y_test == 1).sum()),
        },
        "data_filtering": {
            "signal_columns_used": available_signal_cols,
            "min_signal_features": int(args.min_signal_features),
            "dropped_low_signal_rows": dropped_count,
            "rows_used_for_training_and_eval": int(len(X)),
            "dropped_non_predictive_or_leakage_columns": non_predictive_cols,
        },
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Real", "Fake"],
            output_dict=True,
            zero_division=0,
        ),
    }

    for output_path in [
        args.out_model,
        args.out_predictions,
        args.out_metrics,
        args.out_confusion,
        args.out_feature_importance,
    ]:
        ensure_parent(output_path)

    joblib.dump(pipeline, args.out_model)

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    pred_df["prob_fake"] = y_proba
    pred_df["fake_threshold"] = float(args.fake_threshold)
    pred_df.to_csv(args.out_predictions, index=False)

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Real", "Fake"],
        cmap="Blues",
        colorbar=False,
    )
    disp.ax_.set_title("Confusion Matrix - Deepfake Classifier")
    plt.tight_layout()
    plt.savefig(args.out_confusion, dpi=200)
    plt.close()

    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = extract_feature_names(preprocess)

    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_importances = importances.sort_values(ascending=False).head(15).sort_values()

    plt.figure(figsize=(9, 6))
    top_importances.plot(kind="barh", color="steelblue")
    plt.title("Top 15 Feature Importances - RandomForest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(args.out_feature_importance, dpi=200)
    plt.close()

    print("Deepfake classification pipeline completed.")
    print(f"Input dataset: {args.input}")
    print(
        "Filtering -> "
        f"Dropped low-signal rows: {dropped_count} | "
        f"Rows used: {len(X)}"
    )
    print(f"Features used: {len(numeric_cols)} numeric + {len(categorical_cols)} categorical")
    print(f"Train/Test: {len(X_train)}/{len(X_test)}")
    print(
        "Metrics -> "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )
    print(f"Model: {args.out_model}")
    print(f"Predictions: {args.out_predictions}")
    print(f"Metrics JSON: {args.out_metrics}")
    print(f"Confusion matrix: {args.out_confusion}")
    print(f"Feature importance plot: {args.out_feature_importance}")


if __name__ == "__main__":
    main()
