#!/usr/bin/env python3
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply PCA to normalized deepfake numeric features."
    )
    parser.add_argument(
        "--input",
        default="data/processed/deepfake_dataset_cleaned.csv",
        help="Input cleaned/scaled dataset path.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="Target cumulative explained variance (0-1).",
    )
    parser.add_argument(
        "--out-dataset",
        default="data/processed/deepfake_dataset_pca.csv",
        help="Output transformed dataset path.",
    )
    parser.add_argument(
        "--out-summary",
        default="data/processed/pca_variance_summary.csv",
        help="Output PCA variance summary CSV path.",
    )
    parser.add_argument(
        "--out-plot",
        default="assets/pca_scree_plot.png",
        help="Output scree plot image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.variance_threshold <= 1.0):
        raise ValueError("--variance-threshold must be in (0, 1].")

    df = pd.read_csv(args.input)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # media_id is an identifier, not a predictive numeric feature.
    if "media_id" in numeric_cols:
        numeric_cols.remove("media_id")

    if not numeric_cols:
        raise ValueError("No numeric features available for PCA.")

    X = df[numeric_cols].values

    # Checklist item: covariance matrix and eigen decomposition.
    covariance_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    pca = PCA(n_components=args.variance_threshold)
    X_pca = pca.fit_transform(X)

    component_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=component_names)

    for col in ["label", "media_type", "content_category", "audio_present", "source_platform"]:
        if col in df.columns:
            df_pca[col] = df[col]

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    summary = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    )

    os.makedirs(os.path.dirname(args.out_dataset), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)

    df_pca.to_csv(args.out_dataset, index=False)
    summary.to_csv(args.out_summary, index=False)

    plt.figure(figsize=(8, 5))
    x = np.arange(1, len(cumulative) + 1)
    plt.plot(x, cumulative, marker="o", color="steelblue")
    plt.axhline(y=args.variance_threshold, color="crimson", linestyle="--", linewidth=1)
    plt.xticks(x)
    plt.ylim(0, 1.01)
    plt.xlabel("Numero de Componentes")
    plt.ylabel("Variancia Explicada Cumulativa")
    plt.title("Scree Plot - Variancia Explicada Cumulativa (PCA)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=200)
    plt.close()

    print("PCA pipeline completed.")
    print(f"Input: {args.input}")
    print(f"Numeric features used: {len(numeric_cols)} -> {numeric_cols}")
    print(f"Covariance matrix shape: {covariance_matrix.shape}")
    print(f"Top 3 eigenvalues: {[round(v, 6) for v in eigenvalues[:3]]}")
    print(f"Top eigenvector matrix shape: {eigenvectors.shape}")
    print(f"Variance threshold: {args.variance_threshold}")
    print(f"Components retained: {pca.n_components_}")
    print(f"Cumulative explained variance: {cumulative[-1]:.6f}")
    print(f"PCA dataset: {args.out_dataset}")
    print(f"Variance summary: {args.out_summary}")
    print(f"Scree plot: {args.out_plot}")


if __name__ == "__main__":
    main()
