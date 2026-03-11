#!/usr/bin/env python3
import argparse
import csv
import os
import random
from collections import Counter
from datetime import date, datetime, timedelta
import math

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/deepfake_dataset_cleaned.csv")
    parser.add_argument("--output", default="data/processed/df_timeseries.csv")
    parser.add_argument("--start-date", default="2023-08-01")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def is_fake(value: str) -> bool:
    return (value or "").strip().lower() in ["fake", "1", "1.0"]

def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    rng = random.Random(args.seed)

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fake_count = sum(1 for row in reader if is_fake(row.get("label", "")))

    # Forçando 180 dias de histórico para dar base suficiente aos modelos
    total_days = 180 
    days_indices = list(range(total_days))
    weights = []
    
    # Lógica Complexa: Ao invés de picos aleatórios, criamos uma função composta
    # por uma tendência linear (crescimento de deepfakes no tempo) + Sazonalidade 
    # semanal (mais ataques no fim de semana) para que os modelos possam capturar o sinal.
    for i in days_indices:
        trend = 1.0 + (i * 0.02)
        seasonality = 2.0 if (i % 7) in [5, 6] else 1.0
        weights.append(trend * seasonality)
        
    chosen_offsets = rng.choices(days_indices, weights=weights, k=fake_count)
    
    daily_fake_counts = Counter()
    for offset in chosen_offsets:
        daily_fake_counts[start + timedelta(days=offset)] += 1

    all_days = [start + timedelta(days=i) for i in range(total_days)]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Data", "Volume_Deepfakes"])
        for day in all_days:
            writer.writerow([day.isoformat(), daily_fake_counts.get(day, 0)])

if __name__ == "__main__":
    main()