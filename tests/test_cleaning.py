"""Tests for preprocessing and metadata cleaning."""

import pandas as pd

from src.preprocessing.cleaning import (
    categorize_profession,
    normalize_text_series,
    remove_accents,
)


def test_remove_accents():
    """Accents should be removed correctly."""
    assert remove_accents("école") == "ecole"
    assert remove_accents("République") == "Republique"


def test_categorize_profession_known_categories():
    """Known profession labels should map to expected groups."""
    assert categorize_profession("Professeur de lycée") == "Education"
    assert categorize_profession("Avocat") == "Law"
    assert categorize_profession("Médecin généraliste") == "Health"
    assert categorize_profession("Ingénieur") == "Technical"
    assert categorize_profession("Journaliste") == "Media"


def test_categorize_profession_missing_value():
    """Missing profession should become Unknown."""
    assert categorize_profession(None) == "Unknown"


def test_normalize_text_series():
    """Text normalization should lowercase and remove punctuation."""
    texts = pd.Series(["Bonjour!!! La République, c'est IMPORTANT."])
    normalized = normalize_text_series(texts)

    assert len(normalized) == 1
    assert normalized.iloc[0] == "bonjour la république c est important"
