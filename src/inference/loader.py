"""Utilities to load saved artifacts for inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib


def load_artifacts(experiment_dir: Path) -> dict:
    """Load saved model artifacts from one experiment directory."""
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}")

    lda_path = experiment_dir / "lda_model.joblib"
    vectorizer_path = experiment_dir / "vectorizer.joblib"
    config_path = experiment_dir / "run_config.json"
    labels_path = experiment_dir / "topic_labels.json"

    if not lda_path.exists():
        raise FileNotFoundError(f"Missing artifact: {lda_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Missing artifact: {vectorizer_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing artifact: {config_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing artifact: {labels_path}")

    lda = joblib.load(lda_path)
    vectorizer = joblib.load(vectorizer_path)

    with open(config_path, "r", encoding="utf-8") as file:
        run_config = json.load(file)

    with open(labels_path, "r", encoding="utf-8") as file:
        topic_labels = json.load(file)

    return {
        "lda": lda,
        "vectorizer": vectorizer,
        "run_config": run_config,
        "topic_labels": topic_labels,
    }
