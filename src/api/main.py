"""Bike Sharing prediction API with Evidently and Prometheus monitoring."""

from __future__ import annotations

import datetime as dt
import io
import logging
import time
import zipfile
from typing import Any

import pandas as pd
import requests
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.metrics import MAE, R2Score, RMSE
from evidently.presets import DataDriftPreset
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sklearn.ensemble import RandomForestRegressor

# --- Logging and application configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
TARGET = "cnt"
PREDICTION = "prediction"
DATE_COLUMN = "dteday"
NUM_FEATS = ["temp", "atemp", "hum", "windspeed", "mnth", "hr", "weekday"]
CAT_FEATS = ["season", "holiday", "workingday", "weathersit"]
ALL_MODEL_FEATS = NUM_FEATS + CAT_FEATS

# Keep this registry private to the API, so unrelated process metrics do not leak.
PROMETHEUS_REGISTRY = CollectorRegistry()
API_REQUESTS = Counter(
    "api_requests_total",
    "Total HTTP requests handled by the Bike Sharing API.",
    ["endpoint", "method", "status_code"],
    registry=PROMETHEUS_REGISTRY,
)
API_REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["endpoint", "method", "status_code"],
    registry=PROMETHEUS_REGISTRY,
)
MODEL_RMSE = Gauge(
    "model_rmse_score", "RMSE of the most recent evaluation batch.", registry=PROMETHEUS_REGISTRY
)
MODEL_MAE = Gauge(
    "model_mae_score", "MAE of the most recent evaluation batch.", registry=PROMETHEUS_REGISTRY
)
MODEL_R2 = Gauge(
    "model_r2_score", "R-squared of the most recent evaluation batch.", registry=PROMETHEUS_REGISTRY
)
MODEL_MAPE = Gauge(
    "model_mape_score",
    "MAPE percentage of the most recent evaluation batch (zero targets omitted).",
    registry=PROMETHEUS_REGISTRY,
)
# This gauge is also the basis for the Grafana ML alert and the make fire-alert demo.
# It is useful because input drift can signal a changed operating context before labels arrive.
DATA_DRIFT_DETECTED = Gauge(
    "evidently_data_drift_detected_status",
    "1 when Evidently detects drift in at least one input feature, otherwise 0.",
    registry=PROMETHEUS_REGISTRY,
)
EVALUATED_ITEMS = Gauge(
    "model_evaluated_items", "Number of rows in the most recent evaluation batch.", registry=PROMETHEUS_REGISTRY
)

app = FastAPI(
    title="Bike Sharing Predictor API",
    description="Bike Sharing regression predictions with Evidently and Prometheus monitoring.",
    version="1.0.0",
)

# The model and the January reference batch are built once at process startup.
MODEL: RandomForestRegressor | None = None
REFERENCE_DATA: pd.DataFrame | None = None


class BikeSharingInput(BaseModel):
    """Features for a single Bike Sharing prediction."""

    model_config = ConfigDict(extra="ignore")

    temp: float = Field(..., ge=0, le=1, examples=[0.24])
    atemp: float = Field(..., ge=0, le=1, examples=[0.2879])
    hum: float = Field(..., ge=0, le=1, examples=[0.81])
    windspeed: float = Field(..., ge=0, le=1, examples=[0.0])
    mnth: int = Field(..., ge=1, le=12, examples=[1])
    hr: int = Field(..., ge=0, le=23, examples=[0])
    weekday: int = Field(..., ge=0, le=6, examples=[6])
    season: int = Field(..., ge=1, le=4, examples=[1])
    holiday: int = Field(..., ge=0, le=1, examples=[0])
    workingday: int = Field(..., ge=0, le=1, examples=[0])
    weathersit: int = Field(..., ge=1, le=4, examples=[1])
    dteday: dt.date = Field(..., examples=["2011-01-01"], description="Record date (YYYY-MM-DD).")


class PredictionOutput(BaseModel):
    predicted_count: float


class EvaluationRecord(BikeSharingInput):
    cnt: float = Field(..., ge=0, description="Observed number of rentals.")


class EvaluationData(BaseModel):
    data: list[EvaluationRecord] = Field(..., min_length=1)
    evaluation_period_name: str = Field("unknown_period", min_length=1, max_length=100)


class EvaluationReportOutput(BaseModel):
    message: str
    rmse: float | None
    mape: float | None
    mae: float | None
    r2score: float | None
    drift_detected: int
    evaluated_items: int


def _fetch_data() -> pd.DataFrame:
    """Download the UCI Bike Sharing archive and return its hourly records."""
    logger.info("Fetching the Bike Sharing dataset from UCI")
    try:
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            with archive.open("hour.csv") as csv_file:
                return pd.read_csv(csv_file, parse_dates=[DATE_COLUMN])
    except (requests.RequestException, zipfile.BadZipFile, KeyError) as exc:
        raise RuntimeError(f"Could not load the Bike Sharing dataset: {exc}") from exc


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the UCI table or an API evaluation batch."""
    required = set(ALL_MODEL_FEATS + [DATE_COLUMN])
    missing = sorted(required - set(raw_data.columns))
    if missing:
        raise ValueError(f"Missing required Bike Sharing columns: {', '.join(missing)}")

    data = raw_data.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    if data[DATE_COLUMN].isna().any():
        raise ValueError("dteday must contain valid dates")

    numeric_columns = NUM_FEATS + CAT_FEATS
    if TARGET in data.columns:
        numeric_columns.append(TARGET)
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[numeric_columns].isna().any().any():
        raise ValueError("Bike Sharing features and cnt must be numeric and non-null")

    return data


def _train_and_predict_reference_model(raw_data: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame]:
    """Fit one RandomForestRegressor on January 2011 and return its reference rows."""
    data = _process_data(raw_data)
    if TARGET not in data.columns:
        raise ValueError(f"The training dataset must include {TARGET}")

    january = data.loc[
        (data[DATE_COLUMN].dt.year == 2011) & (data[DATE_COLUMN].dt.month == 1),
        ALL_MODEL_FEATS + [TARGET],
    ].copy()
    if january.empty:
        raise ValueError("The UCI dataset contains no January 2011 records")

    model = RandomForestRegressor(
        n_estimators=120,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(january[ALL_MODEL_FEATS], january[TARGET])
    logger.info("Trained the reference model on %d January 2011 rows", len(january))
    return model, january


def _get_model() -> RandomForestRegressor:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="The reference model is not ready")
    return MODEL


def _metric_number(value: Any) -> float | None:
    """Extract a scalar metric from Evidently's nested Snapshot dictionary."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("current", "value", "mean", "score", "share", "count"):
            if key in value:
                number = _metric_number(value[key])
                if number is not None:
                    return number
        for nested_value in value.values():
            number = _metric_number(nested_value)
            if number is not None:
                return number
    if isinstance(value, list):
        for nested_value in value:
            number = _metric_number(nested_value)
            if number is not None:
                return number
    return None


def _calculate_mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    """Calculate percentage error for rows with non-zero demand only."""
    non_zero_targets = actual != 0
    if not non_zero_targets.any():
        return None
    errors = (actual[non_zero_targets] - predicted[non_zero_targets]).abs()
    percentages = errors / actual[non_zero_targets].abs() * 100
    return float(percentages.mean())


def _extract_report_metric(snapshot: Any, metric_name: str) -> float | None:
    """Read one named metric from the result structure returned by Evidently."""
    report = snapshot.dict()
    target_name = metric_name.casefold()
    for item in report.get("metrics", []):
        metric_id = str(item.get("metric_id") or item.get("metric_name") or "").casefold()
        if target_name in metric_id:
            number = _metric_number(item.get("value", item))
            if number is not None:
                return number
    return None


def _extract_drift_status(snapshot: Any) -> int:
    """Read Evidently's drifted-column summary and convert it to a 0/1 gauge."""
    report = snapshot.dict()
    for item in report.get("metrics", []):
        metric_id = str(item.get("metric_id") or item.get("metric_name") or "").casefold()
        if "driftedcolumnscount" not in metric_id:
            continue
        value = item.get("value", {})
        if isinstance(value, dict):
            for key in ("count", "share", "drifted_columns", "drift_detected"):
                if key in value:
                    number = _metric_number(value[key])
                    if number is not None:
                        return int(number > 0)
        number = _metric_number(value)
        if number is not None:
            return int(number > 0)
    raise RuntimeError("Evidently did not return its DriftedColumnsCount result")


@app.on_event("startup")
def load_reference_model() -> None:
    """Download the reference data and train the model once for this API process."""
    global MODEL, REFERENCE_DATA
    raw_data = _fetch_data()
    MODEL, REFERENCE_DATA = _train_and_predict_reference_model(raw_data)
    reference = REFERENCE_DATA.copy()
    reference[PREDICTION] = MODEL.predict(reference[ALL_MODEL_FEATS])
    REFERENCE_DATA = reference


@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        labels = (endpoint, request.method, str(status_code))
        API_REQUESTS.labels(*labels).inc()
        API_REQUEST_DURATION.labels(*labels).observe(time.perf_counter() - start)


@app.get("/", tags=["service"])
def read_root() -> dict[str, str]:
    return {"message": "Bike Sharing Predictor API. See /docs for endpoints and /metrics for monitoring."}


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
def predict(input_data: BikeSharingInput) -> PredictionOutput:
    model = _get_model()
    feature_row = pd.DataFrame([input_data.model_dump()])[ALL_MODEL_FEATS]
    prediction = float(model.predict(feature_row)[0])
    return PredictionOutput(predicted_count=max(0.0, prediction))


@app.post("/evaluate", response_model=EvaluationReportOutput, tags=["monitoring"])
def evaluate(payload: EvaluationData) -> EvaluationReportOutput:
    model = _get_model()
    if REFERENCE_DATA is None:
        raise HTTPException(status_code=503, detail="The reference data is not ready")

    current = pd.DataFrame([record.model_dump() for record in payload.data])
    try:
        current = _process_data(current)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    current[PREDICTION] = model.predict(current[ALL_MODEL_FEATS])
    reference = REFERENCE_DATA.copy()

    # Evidently uses the declared target/prediction columns for the regression metrics.
    evaluation_definition = DataDefinition(
        numerical_columns=NUM_FEATS + [TARGET, PREDICTION],
        categorical_columns=CAT_FEATS,
        regression=[Regression(target=TARGET, prediction=PREDICTION)],
    )
    current_dataset = Dataset.from_pandas(current[ALL_MODEL_FEATS + [TARGET, PREDICTION]], evaluation_definition)
    reference_dataset = Dataset.from_pandas(reference[ALL_MODEL_FEATS + [TARGET, PREDICTION]], evaluation_definition)
    evaluation_report = Report([RMSE(), MAE(), R2Score()])
    evaluation_snapshot = evaluation_report.run(current_dataset, reference_dataset)

    # Drift compares only the input features. The extra gauge is useful for detecting
    # a changed operating context before labels arrive, and drives the Grafana alert.
    drift_definition = DataDefinition(numerical_columns=NUM_FEATS, categorical_columns=CAT_FEATS)
    current_features = Dataset.from_pandas(current[ALL_MODEL_FEATS], drift_definition)
    reference_features = Dataset.from_pandas(reference[ALL_MODEL_FEATS], drift_definition)
    drift_snapshot = Report([DataDriftPreset(columns=ALL_MODEL_FEATS)]).run(
        current_features, reference_features
    )

    rmse = _extract_report_metric(evaluation_snapshot, "RMSE")
    mae = _extract_report_metric(evaluation_snapshot, "MAE")
    r2score = _extract_report_metric(evaluation_snapshot, "R2Score")
    mape = _calculate_mape(current[TARGET], current[PREDICTION])
    if rmse is None or mae is None or r2score is None:
        logger.error("Could not extract regression scores from Evidently report: %s", evaluation_snapshot.dict())
        raise HTTPException(status_code=500, detail="Evidently did not return all regression scores")
    try:
        drift_detected = _extract_drift_status(drift_snapshot)
    except RuntimeError as exc:
        logger.exception("Could not extract drift score from Evidently report")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    MODEL_RMSE.set(rmse)
    MODEL_MAE.set(mae)
    MODEL_R2.set(r2score)
    if mape is not None:
        MODEL_MAPE.set(mape)
    DATA_DRIFT_DETECTED.set(drift_detected)
    EVALUATED_ITEMS.set(len(current))

    return EvaluationReportOutput(
        message=f"Evaluation completed for {payload.evaluation_period_name}",
        rmse=rmse,
        mape=mape,
        mae=mae,
        r2score=r2score,
        drift_detected=drift_detected,
        evaluated_items=len(current),
    )


@app.post("/trigger-drift", tags=["monitoring"])
def trigger_drift() -> dict[str, Any]:
    """Set the drift gauge to 1 so the provisioned Grafana alert can be verified."""
    DATA_DRIFT_DETECTED.set(1)
    return {
        "message": "Drift alert simulation enabled. Run make evaluation to replace it with a real Evidently result.",
        "evidently_data_drift_detected_status": 1,
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(PROMETHEUS_REGISTRY), media_type=CONTENT_TYPE_LATEST)
