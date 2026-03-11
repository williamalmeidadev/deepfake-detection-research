#!/usr/bin/env python3
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/df_timeseries.csv")
    parser.add_argument("--periods", type=int, default=180) # Argumento agora será respeitado
    parser.add_argument("--out-prophet", default="data/processed/df_prophet.csv")
    parser.add_argument("--out-forecast", default="data/processed/prophet_forecast.csv")
    parser.add_argument("--out-plot", default="assets/prophet_forecast.png")
    parser.add_argument("--out-components", default="assets/prophet_components.png")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    df_ts = pd.read_csv(args.input)

    df_prophet = df_ts.rename(columns={"Data": "ds", "Volume_Deepfakes": "y"}).copy()
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

    model = Prophet()
    model.fit(df_prophet)

    # Lógica Complexa: Substituição do hardcode `periods=14` pelo argumento da CLI
    # Permitindo que o dataframe futuro alinhe sua extensão com o modelo ARIMA no frontend.
    future = model.make_future_dataframe(periods=args.periods, freq="D")
    forecast = model.predict(future)
    
    # Impede previsões negativas de volume no Prophet
    forecast['yhat'] = forecast['yhat'].clip(lower=0)

    os.makedirs(os.path.dirname(args.out_prophet), exist_ok=True)
    df_prophet.to_csv(args.out_prophet, index=False)
    forecast.to_csv(args.out_forecast, index=False)

    print(f"Prophet pipeline completed with {args.periods} future periods.")

if __name__ == "__main__":
    main()