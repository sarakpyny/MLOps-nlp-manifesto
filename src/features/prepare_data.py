"""Dataset preparation functions for manifesto modeling."""

from pathlib import Path

import pandas as pd

from src.data.load_data import load_metadata, merge_metadata_with_texts
from src.preprocessing.cleaning import clean_metadata_columns


def filter_documents(
    df: pd.DataFrame,
    min_doc_length: int,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Filter documents by text length and year range."""
    filtered = df.copy()

    filtered["word_count"] = filtered["text"].astype(str).str.split().str.len()
    filtered = filtered[filtered["word_count"] > min_doc_length].copy()

    filtered["date"] = pd.to_datetime(filtered["date"], errors="coerce")
    filtered["year"] = filtered["date"].dt.year
    filtered = filtered[filtered["year"].between(start_year, end_year)].copy()

    return filtered


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for downstream training and analysis."""
    selected_cols = [
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
    return df[selected_cols].copy()


def load_and_prepare_data(
    metadata_path: Path,
    text_files_path: Path,
    min_doc_length: int,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Load metadata and texts, then prepare a filtered training dataset."""
    metadata = load_metadata(metadata_path)
    df = merge_metadata_with_texts(metadata, text_files_path)
    df = filter_documents(df, min_doc_length, start_year, end_year)
    df = clean_metadata_columns(df)
    return select_columns(df)
