"""Topic modeling utilities for manifesto analysis."""

from __future__ import annotations

from itertools import product

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.data import find
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold


def build_stopwords() -> list[str]:
    """Build the stopword list used by the CountVectorizer."""
    try:
        find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    french_stopwords = set(stopwords.words("french"))

    extra_stopwords = {
        "cevipof", "fonds",
        "circonscription", "elections", "législatives", "tour",
        "candidat", "candidats", "suppléant", "suppléants",
        "maire", "conseiller", "ans",
        "comme", "contre", "faire", "fait", "faut",
        "ceux", "leurs", "depuis", "tout", "tous",
        "être", "falloir", "vouloir", "mettre", "donner",
        "die", "der", "und", "für", "den", "sie", "eine", "das",
        "wir", "werden", "auf", "nicht", "einer", "dass", "gegen",
        "ihr", "auch", "mit", "von", "ist", "dem", "ein", "ich",
        "sich", "wird", "haben", "durch", "ihre", "als",
        "frankreich", "leben", "sind", "mehr", "einen", "politik",
        "mehrheit", "hat", "geben", "juni",
        "alsace", "strasbourg",
        "mai", "juin", "mars", "plus",
        "monsieur", "madame", "mademoiselle",
    }

    return list(french_stopwords.union(extra_stopwords))


def fit_vectorizer_and_lda(
    texts: pd.Series,
    n_topics: int,
    min_df: int,
    max_df: float,
    stop_words: list[str],
    random_seed: int,
) -> tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    """Fit CountVectorizer and LDA on a text corpus."""
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        token_pattern=r"\b[a-zA-ZÀ-ÿ]{3,}\b",
        max_df=max_df,
        min_df=min_df,
    )

    matrix = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_seed,
        learning_method="batch",
    )

    doc_topics = lda.fit_transform(matrix)
    return lda, vectorizer, doc_topics


def train_topic_model(
    texts: pd.Series,
    n_topics: int,
    min_df: int,
    max_df: float,
    stop_words: list[str],
    random_seed: int,
) -> tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    """Train an LDA topic model from processed texts."""
    return fit_vectorizer_and_lda(
        texts=texts,
        n_topics=n_topics,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
        random_seed=random_seed,
    )


def evaluate_lda_perplexity(
    train_texts: pd.Series,
    valid_texts: pd.Series,
    n_topics: int,
    min_df: int,
    max_df: float,
    stop_words: list[str],
    random_seed: int,
) -> float:
    """Train on one fold and return validation perplexity."""
    lda, vectorizer, _ = fit_vectorizer_and_lda(
        texts=train_texts,
        n_topics=n_topics,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
        random_seed=random_seed,
    )

    valid_matrix = vectorizer.transform(valid_texts)
    return float(lda.perplexity(valid_matrix))


def build_candidate_configs(
    candidate_n_topics: list[int],
    candidate_min_df: list[int],
    candidate_max_df: list[float],
) -> list[dict[str, int | float]]:
    """Build the cartesian product of candidate hyperparameters."""
    return [
        {
            "n_topics": n_topics,
            "min_df": min_df,
            "max_df": max_df,
        }
        for n_topics, min_df, max_df in product(
            candidate_n_topics,
            candidate_min_df,
            candidate_max_df,
        )
    ]


def cross_validate_topic_model(
    texts: pd.Series,
    candidate_configs: list[dict[str, int | float]],
    stop_words: list[str],
    random_seed: int,
    cv_folds: int,
) -> pd.DataFrame:
    """Evaluate candidate topic-model configurations with K-fold CV."""
    if cv_folds <= 1:
        rows = []
        for config in candidate_configs:
            lda, vectorizer, _ = fit_vectorizer_and_lda(
                texts=texts,
                n_topics=int(config["n_topics"]),
                min_df=int(config["min_df"]),
                max_df=float(config["max_df"]),
                stop_words=stop_words,
                random_seed=random_seed,
            )
            matrix = vectorizer.transform(texts)
            perplexity = float(lda.perplexity(matrix))
            rows.append(
                {
                    "n_topics": int(config["n_topics"]),
                    "min_df": int(config["min_df"]),
                    "max_df": float(config["max_df"]),
                    "fold": 0,
                    "perplexity": perplexity,
                }
            )
        return pd.DataFrame(rows)

    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    texts = texts.reset_index(drop=True)

    rows: list[dict[str, int | float]] = []

    for config in candidate_configs:
        for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(texts), start=1):
            train_texts = texts.iloc[train_idx]
            valid_texts = texts.iloc[valid_idx]

            perplexity = evaluate_lda_perplexity(
                train_texts=train_texts,
                valid_texts=valid_texts,
                n_topics=int(config["n_topics"]),
                min_df=int(config["min_df"]),
                max_df=float(config["max_df"]),
                stop_words=stop_words,
                random_seed=random_seed,
            )

            rows.append(
                {
                    "n_topics": int(config["n_topics"]),
                    "min_df": int(config["min_df"]),
                    "max_df": float(config["max_df"]),
                    "fold": fold_idx,
                    "perplexity": perplexity,
                }
            )

    return pd.DataFrame(rows)


def summarize_cv_results(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CV results by candidate configuration."""
    summary = (
        cv_results
        .groupby(["n_topics", "min_df", "max_df"], as_index=False)
        .agg(
            mean_perplexity=("perplexity", "mean"),
            std_perplexity=("perplexity", "std"),
            n_folds=("fold", "count"),
        )
        .sort_values("mean_perplexity", ascending=True)
        .reset_index(drop=True)
    )
    return summary


def select_best_config(cv_summary: pd.DataFrame) -> dict[str, int | float]:
    """Return the best configuration according to mean perplexity."""
    if cv_summary.empty:
        raise ValueError("cv_summary is empty.")

    best_row = cv_summary.iloc[0]
    return {
        "n_topics": int(best_row["n_topics"]),
        "min_df": int(best_row["min_df"]),
        "max_df": float(best_row["max_df"]),
    }


def extract_topics(
    model: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extract top words for each topic into a dataframe."""
    feature_names = vectorizer.get_feature_names_out()
    rows = []

    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-top_n:][::-1]
        top_words = [feature_names[index] for index in top_indices]
        rows.append({"topic_id": topic_idx, "top_words": ", ".join(top_words)})

    return pd.DataFrame(rows)


def generate_topic_labels(topics_df: pd.DataFrame) -> dict[str, str]:
    """Generate simple human-readable labels from top words per topic."""
    topic_labels = {}

    for _, row in topics_df.iterrows():
        topic_id = int(row["topic_id"])
        top_words = str(row["top_words"]).split(", ")
        topic_labels[str(topic_id)] = "_".join(top_words[:3])

    return topic_labels
