#!/usr/bin/env python3
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from prophet import Prophet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Prophet model using deepfake daily time-series."
    )
    parser.add_argument(
        "--input",
        default="data/processed/df_timeseries.csv",
        help="Input daily time-series CSV with Data and Volume_Deepfakes columns.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=30,
        help="Number of future days to forecast.",
    )
    parser.add_argument(
        "--out-prophet",
        default="data/processed/df_prophet.csv",
        help="Output CSV formatted for Prophet (ds, y).",
    )
    parser.add_argument(
        "--out-forecast",
        default="data/processed/prophet_forecast.csv",
        help="Output CSV with Prophet forecast results.",
    )
    parser.add_argument(
        "--out-plot",
        default="assets/prophet_forecast.png",
        help="Output forecast plot path.",
    )
    parser.add_argument(
        "--out-components",
        default="assets/prophet_components.png",
        help="Output components plot path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_ts = pd.read_csv(args.input)
    required = {"Data", "Volume_Deepfakes"}
    missing = required - set(df_ts.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df_prophet = df_ts.rename(columns={"Data": "ds", "Volume_Deepfakes": "y"}).copy()
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=args.periods, freq="D")
    forecast = model.predict(future)

    os.makedirs(os.path.dirname(args.out_prophet), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_forecast), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_components), exist_ok=True)

    df_prophet.to_csv(args.out_prophet, index=False)
    forecast.to_csv(args.out_forecast, index=False)

    fig_forecast = model.plot(forecast)
    fig_forecast.savefig(args.out_plot, dpi=200, bbox_inches="tight")
    plt.close(fig_forecast)

    fig_components = model.plot_components(forecast)
    fig_components.savefig(args.out_components, dpi=200, bbox_inches="tight")
    plt.close(fig_components)

    print("Prophet pipeline completed.")
    print(f"Input: {args.input}")
    print(f"Formatted output (ds,y): {args.out_prophet}")
    print(f"Forecast output: {args.out_forecast}")
    print(f"Forecast plot: {args.out_plot}")
    print(f"Components plot: {args.out_components}")


if __name__ == "__main__":
    main()
