"""Fetch Bike Sharing records, evaluate an API batch, and optionally simulate traffic."""

import argparse
import io
import os
import sys
import zipfile

import pandas as pd
import requests

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
API_EVALUATE_URL = os.getenv("API_EVALUATE_URL", "http://bike-api:8080/evaluate")
API_PREDICT_URL = os.getenv("API_PREDICT_URL", "http://bike-api:8080/predict")
EVALUATION_SAMPLE_SIZE = 1000
WEEKLY_PERIODS = {
    "week1_february": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week2_february": ("2011-02-08 00:00:00", "2011-02-14 23:00:00"),
    "week3_february": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}
DEFAULT_PERIOD_NAME = "week1_february"
NUM_FEATS = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CAT_FEATS = ["season", "holiday", "workingday", "weathersit"]
ALL_MODEL_FEATS = NUM_FEATS + CAT_FEATS
TARGET = "cnt"
DATE_COLUMN = "dteday"


def _fetch_data(dataset_url: str = DATASET_URL) -> pd.DataFrame:
    """Download the UCI Bike Sharing archive and read its hourly CSV."""
    print(f"Fetching Bike Sharing dataset from {dataset_url}...")
    response = requests.get(dataset_url, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        with archive.open("hour.csv") as csv_file:
            return pd.read_csv(csv_file, parse_dates=[DATE_COLUMN])


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Build a chronological hourly index used for date-range selection."""
    required = set(ALL_MODEL_FEATS + [TARGET, DATE_COLUMN])
    missing = sorted(required - set(raw_data.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    data = raw_data.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="raise").dt.normalize()
    data["hr"] = data["hr"].astype(int)
    data.index = data[DATE_COLUMN] + pd.to_timedelta(data["hr"], unit="h")
    return data.sort_index()


def run_evaluation(
    full_data: pd.DataFrame,
    period_name: str,
    start_date_str: str,
    end_date_str: str,
    api_url: str = API_EVALUATE_URL,
) -> dict:
    """Send the selected labeled period to the API and return its report."""
    print(f"\n--- Evaluating {period_name} ({start_date_str} to {end_date_str}) ---")
    current_period_data = full_data.loc[start_date_str:end_date_str].copy()
    if current_period_data.empty:
        raise ValueError(f"No Bike Sharing records found for {period_name}")

    columns = ALL_MODEL_FEATS + [TARGET, DATE_COLUMN]
    eval_rows = current_period_data[columns]
    if len(eval_rows) > EVALUATION_SAMPLE_SIZE:
        eval_rows = eval_rows.sample(n=EVALUATION_SAMPLE_SIZE, random_state=42)
    eval_rows = eval_rows.copy()
    eval_rows[DATE_COLUMN] = eval_rows[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    payload = {
        "data": eval_rows.to_dict(orient="records"),
        "evaluation_period_name": period_name,
    }

    response = requests.post(api_url, json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()
    print("Evaluation successful:")
    for label, key in (("RMSE", "rmse"), ("MAE", "mae"), ("R2", "r2score"), ("MAPE", "mape")):
        value = result.get(key)
        print(f"  - {label}: {value:.4f}" if value is not None else f"  - {label}: N/A")
    print(f"  - Data drift detected: {'yes' if result.get('drift_detected') else 'no'}")
    print(f"  - Evaluated rows: {result.get('evaluated_items')}")
    return result


def generate_traffic(
    count: int,
    full_data: pd.DataFrame,
    api_url: str = API_PREDICT_URL,
) -> int:
    """Send representative January prediction requests to the API."""
    if count < 0:
        raise ValueError("Traffic count cannot be negative")
    if count == 0:
        return 0

    prediction_data = full_data.loc["2011-01-01 00:00:00":"2011-01-31 23:00:00"]
    if prediction_data.empty:
        raise ValueError("No January records are available for prediction traffic")
    sample_count = min(count, len(prediction_data))
    samples = prediction_data.sample(n=sample_count, random_state=42)
    print(f"\n--- Sending {sample_count} prediction requests to {api_url} ---")

    for index, (_, row) in enumerate(samples.iterrows(), start=1):
        request_data = {column: row[column] for column in ALL_MODEL_FEATS}
        request_data[DATE_COLUMN] = row[DATE_COLUMN].strftime("%Y-%m-%d")
        response = requests.post(api_url, json=request_data, timeout=15)
        response.raise_for_status()
        if index % 10 == 0 or index == sample_count:
            print(f"  - Sent {index}/{sample_count}")
    return sample_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", choices=sorted(WEEKLY_PERIODS), default=DEFAULT_PERIOD_NAME)
    parser.add_argument("--traffic-count", type=int, default=0)
    parser.add_argument("--evaluate-url", default=API_EVALUATE_URL)
    parser.add_argument("--predict-url", default=API_PREDICT_URL)
    parser.add_argument("--dataset-url", default=DATASET_URL)
    args = parser.parse_args()

    try:
        full_data = _process_data(_fetch_data(args.dataset_url))
        start_date, end_date = WEEKLY_PERIODS[args.period]
        run_evaluation(full_data, args.period, start_date, end_date, args.evaluate_url)
        generate_traffic(args.traffic_count, full_data, args.predict_url)
    except (requests.RequestException, zipfile.BadZipFile, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
