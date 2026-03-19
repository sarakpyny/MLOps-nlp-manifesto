"""Inference utilities for manifesto topic prediction."""

from __future__ import annotations

from pathlib import Path

from src.inference.loader import load_artifacts
from src.preprocessing.cleaning import build_processed_text


def predict_topics(text: str, experiment_dir: Path) -> dict:
    """Predict the topic mixture for one raw text using saved artifacts."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    artifacts = load_artifacts(experiment_dir)

    run_config = artifacts["run_config"]
    processed_text = build_processed_text(
        text=text,
        use_lemmatization=run_config["use_lemmatization"],
        spacy_model=run_config["resolved_spacy_model"],
    )

    vectorized = artifacts["vectorizer"].transform([processed_text])
    topic_distribution = artifacts["lda"].transform(vectorized)[0]

    top_topic_id = int(topic_distribution.argmax())
    top_topic_score = float(topic_distribution[top_topic_id])
    top_topic_label = artifacts["topic_labels"].get(
        str(top_topic_id),
        f"topic_{top_topic_id}",
    )

    return {
        "processed_text": processed_text,
        "top_topic_id": top_topic_id,
        "top_topic_label": top_topic_label,
        "top_topic_score": top_topic_score,
        "topic_distribution": topic_distribution.tolist(),
    }
