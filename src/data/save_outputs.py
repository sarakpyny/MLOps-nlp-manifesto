"""Saving utilities for trained artifacts and experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def save_run_config(args: argparse.Namespace, run_dir: Path, spacy_model: str) -> None:
    """Save training configuration as JSON."""
    config = vars(args).copy()
    config["resolved_spacy_model"] = spacy_model
    pd.Series(config).to_json(run_dir / "run_config.json", indent=2)


def save_topic_labels(topic_labels: dict[str, str], run_dir: Path) -> None:
    """Save topic labels as JSON."""
    with open(run_dir / "topic_labels.json", "w", encoding="utf-8") as file:
        json.dump(topic_labels, file, ensure_ascii=False, indent=2)


def save_outputs(
    df: pd.DataFrame,
    doc_topics: np.ndarray,
    topics_df: pd.DataFrame,
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    output_dir: Path,
    experiment_name: str,
    args: argparse.Namespace,
    spacy_model: str,
    topic_labels: dict[str, str],
) -> None:
    """Save experiment outputs, topic tables, trained artifacts, and topic labels."""
    run_dir = output_dir / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    doc_topics_df = pd.DataFrame(
        doc_topics,
        columns=[f"Topic_{index}" for index in range(doc_topics.shape[1])],
    )

    result_df = pd.concat([df.reset_index(drop=True), doc_topics_df], axis=1)
    result_df["dominant_topic"] = doc_topics_df.idxmax(axis=1)

    result_df.to_csv(
        run_dir / "data_topics.csv",
        index=False,
        encoding="utf-8",
    )
    topics_df.to_csv(
        run_dir / "topics_summary.csv",
        index=False,
        encoding="utf-8",
    )

    joblib.dump(lda, run_dir / "lda_model.joblib")
    joblib.dump(vectorizer, run_dir / "vectorizer.joblib")

    save_run_config(args, run_dir, spacy_model)
    save_topic_labels(topic_labels, run_dir)
