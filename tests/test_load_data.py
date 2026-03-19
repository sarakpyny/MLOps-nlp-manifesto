"""Tests for raw data loading and merge helpers."""

from pathlib import Path

import pandas as pd

from src.data.load_data import load_text_files, merge_metadata_with_texts


def test_load_text_files_reads_txt_files(tmp_path: Path):
    """Text loader should read .txt files and use filename stem as key."""
    file_a = tmp_path / "123.txt"
    file_b = tmp_path / "456.txt"

    file_a.write_text("texte un", encoding="utf-8")
    file_b.write_text("texte deux", encoding="utf-8")

    text_dict = load_text_files(tmp_path)

    assert text_dict == {
        "123": "texte un",
        "456": "texte deux",
    }


def test_merge_metadata_with_texts_keeps_only_matching_rows(tmp_path: Path):
    """Merge should keep only metadata rows with matching text files."""
    (tmp_path / "1.txt").write_text("manifeste 1", encoding="utf-8")
    (tmp_path / "2.txt").write_text("manifeste 2", encoding="utf-8")

    metadata = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["1981-01-01", "1982-01-01", "1983-01-01"],
        }
    )

    merged = merge_metadata_with_texts(metadata, tmp_path)

    assert merged["id"].tolist() == [1, 2]
    assert merged["text"].tolist() == ["manifeste 1", "manifeste 2"]
