"""Tests for topic modeling utilities."""

import pandas as pd

from src.models.topic_model import extract_topics, train_topic_model


def test_train_topic_model_returns_expected_shape():
    """Document-topic matrix should have shape (n_docs, n_topics)."""
    texts = pd.Series(
        [
            "republique democratie justice sociale travail",
            "agriculture economie territoire nation industrie",
            "education universite jeunesse savoir recherche",
        ]
    )

    n_topics = 2

    lda, vectorizer, doc_topics = train_topic_model(
        texts=texts,
        n_topics=n_topics,
        min_df=1,
        max_df=1.0,
        stop_words=[],
        random_seed=42,
    )

    assert doc_topics.shape == (3, 2)
    assert lda.n_components == 2
    assert len(vectorizer.get_feature_names_out()) > 0


def test_extract_topics_returns_expected_columns():
    """Extracted topics should contain topic ids and top words."""
    texts = pd.Series(
        [
            "republique democratie justice sociale travail",
            "agriculture economie territoire nation industrie",
            "education universite jeunesse savoir recherche",
        ]
    )

    lda, vectorizer, _ = train_topic_model(
        texts=texts,
        n_topics=2,
        min_df=1,
        max_df=1.0,
        stop_words=[],
        random_seed=42,
    )

    topics_df = extract_topics(model=lda, vectorizer=vectorizer, top_n=3)

    assert list(topics_df.columns) == ["topic_id", "top_words"]
    assert len(topics_df) == 2
    assert topics_df["top_words"].str.len().gt(0).all()
