"""Tests for inference utilities."""

from pathlib import Path

import pytest

from src.inference.predictor import predict_topics


def test_predict_topics_raises_on_empty_text():
    """Empty input text should raise a ValueError."""
    with pytest.raises(ValueError):
        predict_topics(
            text="",
            experiment_dir=Path("outputs/baseline_lda"),
        )


def test_predict_topics_returns_expected_keys():
    """Prediction output should contain the expected fields."""
    result = predict_topics(
        text="Nous défendons la justice sociale et les services publics.",
        experiment_dir=Path("outputs/baseline_lda"),
    )

    assert "processed_text" in result
    assert "top_topic_id" in result
    assert "top_topic_label" in result
    assert "top_topic_score" in result
    assert "topic_distribution" in result
    assert isinstance(result["topic_distribution"], list)


def test_topic_distribution_sums_to_one():
    """Topic probabilities should sum approximately to one."""
    result = predict_topics(
        text="Nous défendons la justice sociale et les services publics.",
        experiment_dir=Path("outputs/baseline_lda"),

    )

    assert abs(sum(result["topic_distribution"]) - 1.0) < 1e-6
