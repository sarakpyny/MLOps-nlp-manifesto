"""FastAPI application for manifesto topic inference and analysis."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_NAME = "Manifesto Topic API"
APP_VERSION = "0.3.0"
DEFAULT_EXPERIMENT_DIR = Path("outputs/baseline_lda")
DEFAULT_REGISTERED_MODEL = "manifesto_topic_model"
DEFAULT_MODEL_ALIAS = "production"

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    """Request body for topic prediction."""

    text: str = Field(
        ...,
        min_length=1,
        description="Raw manifesto text to analyze.",
    )


class PredictResponse(BaseModel):
    """Response body for topic prediction."""

    processed_text: str
    top_topic_id: int
    top_topic_label: str
    top_topic_score: float
    topic_distribution: list[float]


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API for French electoral manifesto topic inference and analytical summaries.",
)


def get_experiment_dir() -> Path:
    """Return the default experiment directory for local analytical files."""
    return DEFAULT_EXPERIMENT_DIR


def get_tracking_uri() -> str | None:
    """Return the MLflow tracking URI from environment."""
    return os.getenv("MLFLOW_TRACKING_URI")


def get_registered_model_name() -> str:
    """Return the MLflow registered model name."""
    return os.getenv("MLFLOW_MODEL_NAME", DEFAULT_REGISTERED_MODEL)


def get_model_uri() -> str:
    """Build the MLflow model URI."""
    model_name = get_registered_model_name()
    model_alias = os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_MODEL_ALIAS)
    return f"models:/{model_name}@{model_alias}"


def use_registry_backend() -> bool:
    """Return whether registry-based prediction is enabled."""
    return bool(get_tracking_uri())


@lru_cache(maxsize=1)
def load_registry_model():
    """Load and cache the prediction model from MLflow registry."""
    tracking_uri = get_tracking_uri()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    return mlflow.pyfunc.load_model(get_model_uri())


def predict_topics_local(text: str) -> dict[str, Any]:
    """Predict topics using saved local artifacts."""
    from src.inference.predictor import predict_topics as local_predict_topics

    return local_predict_topics(
        text=text,
        experiment_dir=get_experiment_dir(),
    )


def predict_topics(text: str) -> dict[str, Any]:
    """Predict topics using registry if available, otherwise local artifacts."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    if use_registry_backend():
        try:
            model = load_registry_model()
            input_df = pd.DataFrame({"text": [text]})
            predictions = model.predict(input_df)

            if len(predictions) == 0:
                raise ValueError("Model returned no prediction.")

            result = predictions[0]
            if not isinstance(result, dict):
                raise ValueError(
                    "Model returned an unexpected prediction format.")

            return result
        except Exception as exc:
            logger.warning(
                "Registry prediction failed, falling back to local artifacts: %s",
                exc,
            )

    return predict_topics_local(text)


def read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv_file(path: Path) -> pd.DataFrame:
    """Read a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def get_topic_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted topic columns safely."""
    topic_cols: list[str] = []

    for col in df.columns:
        if col.startswith("Topic_"):
            try:
                int(col.split("_")[1])
                topic_cols.append(col)
            except (IndexError, ValueError):
                continue

    return sorted(topic_cols, key=lambda col: int(col.split("_")[1]))


def get_topic_label_map(experiment_dir: Path) -> dict[str, str]:
    """Load topic labels from the saved experiment."""
    return read_json_file(experiment_dir / "topic_labels.json")


def get_topics_summary_df(experiment_dir: Path) -> pd.DataFrame:
    """Load the topic summary CSV."""
    return read_csv_file(experiment_dir / "topics_summary.csv")


def get_data_topics_df(experiment_dir: Path) -> pd.DataFrame:
    """Load the document-topic analysis CSV."""
    return read_csv_file(experiment_dir / "data_topics.csv")


@app.get("/")
def root() -> dict[str, str]:
    """Welcome endpoint."""
    return {
        "message": "Manifesto Topic API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint."""
    experiment_dir = get_experiment_dir()

    required_local_files = [
        experiment_dir / "topic_labels.json",
        experiment_dir / "topics_summary.csv",
        experiment_dir / "data_topics.csv",
    ]
    missing_local_files = [str(path)
                           for path in required_local_files if not path.exists()]

    registry_status = None
    registry_error = None

    if use_registry_backend():
        registry_status = "ok"
        try:
            load_registry_model()
        except Exception as exc:  # pragma: no cover
            registry_status = "degraded"
            registry_error = str(exc)

    overall_status = "ok"
    if missing_local_files:
        overall_status = "degraded"
    if use_registry_backend() and registry_error:
        overall_status = "degraded"

    return {
        "status": overall_status,
        "prediction_backend": "mlflow_registry" if use_registry_backend() else "local_artifacts",
        "model_uri": get_model_uri() if use_registry_backend() else None,
        "tracking_uri": get_tracking_uri(),
        "registry_status": registry_status,
        "registry_error": registry_error,
        "analysis_experiment_dir": str(experiment_dir),
        "missing_local_files": missing_local_files,
    }


@app.post("/predict_topics", response_model=PredictResponse)
def predict_topics_endpoint(payload: PredictRequest) -> PredictResponse:
    """Predict topic distribution for one text."""
    try:
        result = predict_topics(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(**result)


@app.get("/topics")
def get_topics() -> dict[str, list[dict[str, Any]]]:
    """Return saved topic labels and top words."""
    experiment_dir = get_experiment_dir()

    try:
        topic_labels = get_topic_label_map(experiment_dir)
        topics_summary = get_topics_summary_df(experiment_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    topics: list[dict[str, Any]] = []

    for _, row in topics_summary.iterrows():
        topic_id = int(row["topic_id"])
        topics.append(
            {
                "topic_id": topic_id,
                "label": topic_labels.get(str(topic_id), f"topic_{topic_id}"),
                "top_words": row["top_words"],
            }
        )

    return {"topics": topics}


@app.get("/stats")
def get_stats() -> dict[str, Any]:
    """Return dataset-level descriptive statistics."""
    experiment_dir = get_experiment_dir()

    try:
        df = get_data_topics_df(experiment_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        stats: dict[str, Any] = {
            "n_documents": int(len(df)),
            "n_topics": int(len(get_topic_columns(df))),
            "n_parties": int(df["party_clean"].nunique()) if "party_clean" in df.columns else None,
            "n_professions": int(df["profession_clean"].nunique()) if "profession_clean" in df.columns else None,
            "year_min": int(df["year"].min()) if "year" in df.columns else None,
            "year_max": int(df["year"].max()) if "year" in df.columns else None,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Stats failed: {exc}") from exc

    if "dominant_topic" in df.columns:
        dominant_counts = df["dominant_topic"].value_counts(
        ).sort_index().to_dict()
        stats["dominant_topic_counts"] = {}

        for key, value in dominant_counts.items():
            if isinstance(key, str) and key.startswith("Topic_"):
                topic_id = key.split("_")[1]
            else:
                topic_id = str(key)

            stats["dominant_topic_counts"][topic_id] = int(value)

    return stats


def build_profile(
    df: pd.DataFrame,
    group_column: str,
    group_value: str,
    topic_labels: dict[str, str],
) -> dict[str, Any]:
    """Build a profile for one party or one profession."""
    filtered_df = df[df[group_column].astype(
        str).str.lower() == group_value.lower()].copy()

    if filtered_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No records found for {group_column}='{group_value}'.",
        )

    topic_cols = get_topic_columns(filtered_df)
    try:
        mean_scores = filtered_df[topic_cols].mean(
        ).sort_values(ascending=False)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Profile computation failed: {exc}",
        ) from exc

    top_topics: list[dict[str, Any]] = []
    for topic_col, score in mean_scores.head(5).items():
        topic_id = int(topic_col.split("_")[1])
        top_topics.append(
            {
                "topic_id": topic_id,
                "label": topic_labels.get(str(topic_id), f"topic_{topic_id}"),
                "mean_score": float(score),
            }
        )

    dominant_distribution = (
        filtered_df["dominant_topic"].value_counts().sort_index().to_dict()
        if "dominant_topic" in filtered_df.columns
        else {}
    )

    dominant_topic_counts = {}
    for key, value in dominant_distribution.items():
        if isinstance(key, str) and key.startswith("Topic_"):
            topic_id = key.split("_")[1]
        else:
            topic_id = str(key)

        dominant_topic_counts[topic_id] = int(value)

    return {
        "group_column": group_column,
        "group_value": group_value,
        "n_documents": int(len(filtered_df)),
        "top_topics": top_topics,
        "dominant_topic_counts": dominant_topic_counts,
    }


@app.get("/party_profile/{party}")
def get_party_profile(party: str) -> dict[str, Any]:
    """Return topic profile for one party."""
    experiment_dir = get_experiment_dir()

    try:
        df = get_data_topics_df(experiment_dir)
        topic_labels = get_topic_label_map(experiment_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "party_clean" not in df.columns:
        raise HTTPException(
            status_code=500, detail="Column 'party_clean' not found.")

    return build_profile(
        df=df,
        group_column="party_clean",
        group_value=party,
        topic_labels=topic_labels,
    )


@app.get("/profession_profile/{profession}")
def get_profession_profile(profession: str) -> dict[str, Any]:
    """Return topic profile for one profession."""
    experiment_dir = get_experiment_dir()

    try:
        df = get_data_topics_df(experiment_dir)
        topic_labels = get_topic_label_map(experiment_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if "profession_clean" not in df.columns:
        raise HTTPException(
            status_code=500, detail="Column 'profession_clean' not found.")

    return build_profile(
        df=df,
        group_column="profession_clean",
        group_value=profession,
        topic_labels=topic_labels,
    )
