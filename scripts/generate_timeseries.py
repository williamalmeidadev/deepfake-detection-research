#!/usr/bin/env python3
import argparse
import csv
import os
import random
from collections import Counter
from datetime import date, datetime, timedelta


DATE_CANDIDATES = {
    "date",
    "data",
    "datetime",
    "timestamp",
    "created_at",
    "event_time",
    "event_date",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a daily deepfake time-series from a tabular dataset."
    )
    parser.add_argument(
        "--input",
        default="data/processed/deepfake_dataset_cleaned.csv",
        help="Input CSV with at least a label column.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/df_timeseries.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="Start date for simulated distribution (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Simulation period in years (used when no date column exists).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic simulation.",
    )
    return parser.parse_args()


def find_date_column(fieldnames: list[str]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in DATE_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def parse_date(value: str) -> date:
    value = value.strip()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()


def is_fake(value: str) -> bool:
    return (value or "").strip().lower() == "fake"


def add_years(base: date, years: int) -> date:
    try:
        return base.replace(year=base.year + years)
    except ValueError:
        # Handles Feb 29 -> Feb 28 on non-leap years.
        return base.replace(month=2, day=28, year=base.year + years)


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    end_exclusive = add_years(start, args.years)
    total_days = (end_exclusive - start).days

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")

        date_col = find_date_column(reader.fieldnames)
        if "label" not in reader.fieldnames:
            raise ValueError("Input CSV must contain 'label' column.")

        rows = list(reader)

    daily_fake_counts: Counter[date] = Counter()

    if date_col:
        for row in rows:
            if not is_fake(row.get("label", "")):
                continue
            row_date = parse_date(row.get(date_col, ""))
            daily_fake_counts[row_date] += 1
        mode = f"using existing date column '{date_col}'"
    else:
        rng = random.Random(args.seed)
        for row in rows:
            if not is_fake(row.get("label", "")):
                continue
            offset = rng.randint(0, total_days - 1)
            row_date = start + timedelta(days=offset)
            daily_fake_counts[row_date] += 1
        mode = "simulated dates (no date column found)"

    all_days = [start + timedelta(days=i) for i in range(total_days)]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Data", "Volume_Deepfakes"])
        for day in all_days:
            writer.writerow([day.isoformat(), daily_fake_counts.get(day, 0)])

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode: {mode}")
    print(f"Days generated: {len(all_days)}")
    print(f"Total fake records counted: {sum(daily_fake_counts.values())}")


if __name__ == "__main__":
    main()
