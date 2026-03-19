"""Topic modeling utilities for manifesto analysis."""

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def build_stopwords() -> list[str]:
    """Build the stopword list used by the CountVectorizer."""
    french_stopwords = set(stopwords.words("french"))

    extra_stopwords = {
        "cevipof",
        "fonds",
        "circonscription",
        "elections",
        "législatives",
        "tour",
        "candidat",
        "candidats",
        "suppléant",
        "suppléants",
        "maire",
        "conseiller",
        "ans",
        "comme",
        "contre",
        "faire",
        "fait",
        "faut",
        "ceux",
        "leurs",
        "depuis",
        "tout",
        "tous",
        "être",
        "falloir",
        "vouloir",
        "mettre",
        "donner",
        "die",
        "der",
        "und",
        "für",
        "den",
        "sie",
        "eine",
        "das",
        "wir",
        "werden",
        "auf",
        "nicht",
        "einer",
        "dass",
        "gegen",
        "ihr",
        "auch",
        "mit",
        "von",
        "ist",
        "dem",
        "ein",
        "ich",
        "sich",
        "wird",
        "haben",
        "durch",
        "ihre",
        "als",
        "frankreich",
        "leben",
        "sind",
        "mehr",
        "einen",
        "politik",
        "mehrheit",
        "hat",
        "geben",
        "juni",
        "alsace",
        "strasbourg",
    }

    return list(french_stopwords.union(extra_stopwords))


def train_topic_model(
    texts: pd.Series,
    n_topics: int,
    min_df: int,
    max_df: float,
    stop_words: list[str],
    random_seed: int,
) -> tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    """Train an LDA topic model from processed texts."""
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
