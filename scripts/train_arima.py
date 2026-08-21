#!/usr/bin/env python3
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ARIMA model using deepfake daily time-series."
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
        "--out-forecast",
        default="data/processed/arima_forecast.csv",
        help="Output CSV with ARIMA forecast results.",
    )
    parser.add_argument(
        "--out-plot",
        default="assets/arima_forecast.png",
        help="Output forecast plot path.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    df_ts = pd.read_csv(args.input)
    required = {"Data", "Volume_Deepfakes"}
    missing = required - set(df_ts.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    y_series = pd.to_numeric(df_ts["Volume_Deepfakes"], errors="coerce").fillna(0).astype("float64")
    dates = pd.to_datetime(df_ts["Data"])

    arima_model = ARIMA(y_series, order=(1, 1, 1))
    arima_fit = arima_model.fit()

    # Forecast
    arima_values = arima_fit.forecast(steps=args.periods)
    last_date = dates.max()
    arima_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=args.periods, freq="D"
    )

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({"ds": arima_dates, "yhat": arima_values})

    os.makedirs(os.path.dirname(args.out_forecast), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_plot), exist_ok=True)

    forecast_df.to_csv(args.out_forecast, index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y_series, label="Dados Reais", color="blue")
    plt.plot(arima_dates, arima_values, label="Previsão ARIMA", color="red", linestyle="--")
    plt.title("Previsão ARIMA para Volume de Deepfakes")
    plt.xlabel("Data")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.out_plot, dpi=200, bbox_inches="tight")
    plt.close()

    print("ARIMA pipeline completed.")
    print(f"Input: {args.input}")
    print(f"Forecast output: {args.out_forecast}")
    print(f"Forecast plot: {args.out_plot}")

if __name__ == "__main__":
    main()
