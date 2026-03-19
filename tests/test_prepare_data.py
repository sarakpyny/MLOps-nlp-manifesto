"""Tests for dataset preparation helpers."""

import pandas as pd

from src.features.prepare_data import filter_documents, select_columns


def test_filter_documents_removes_short_and_out_of_range_docs():
    """Documents should be filtered by word count and year range."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": [
                "mot " * 120,   # long enough, year ok
                "court texte",  # too short
                "mot " * 150,   # long enough, but year out of range
            ],
            "date": ["1985-01-01", "1985-01-01", "2000-01-01"],
        }
    )

    filtered = filter_documents(
        df=df,
        min_doc_length=100,
        start_year=1981,
        end_year=1993,
    )

    assert filtered["id"].tolist() == [1]
    assert filtered["year"].tolist() == [1985]
    assert filtered["word_count"].tolist() == [120]


def test_select_columns_keeps_expected_schema():
    """Column selection should keep only downstream-required columns."""
    df = pd.DataFrame(
        {
            "id": [1],
            "date": ["1985-01-01"],
            "year": [1985],
            "titulaire-liste": ["PS"],
            "titulaire-profession": ["Professeur"],
            "titulaire-age-calcule": [45],
            "titulaire-sexe": ["F"],
            "departement-insee": ["75"],
            "text": ["texte"],
            "word_count": [100],
            "profession_clean": ["Education"],
            "party_clean": ["Other"],
            "extra_column": ["drop me"],
        }
    )

    selected = select_columns(df)

    expected_columns = [
        "id",
        "date",
        "year",
        "titulaire-liste",
        "titulaire-profession",
        "titulaire-age-calcule",
        "titulaire-sexe",
        "departement-insee",
        "text",
        "word_count",
        "profession_clean",
        "party_clean",
    ]

    assert list(selected.columns) == expected_columns
