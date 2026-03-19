"""Utilities for loading raw manifesto data and text files."""

import os
import zipfile
from pathlib import Path

import pandas as pd


def unzip_archives(base_path: Path) -> None:
    """Extract zip archives found under the given directory."""
    for root, _, files in os.walk(base_path):
        for file_name in files:
            if not file_name.endswith(".zip"):
                continue

            zip_path = Path(root) / file_name
            extract_folder = Path(root) / Path(file_name).stem

            if extract_folder.exists() and any(extract_folder.iterdir()):
                continue

            extract_folder.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_folder)
            except zipfile.BadZipFile:
                print(f"Bad zip file skipped: {zip_path}")


def load_text_files(base_path: Path) -> dict[str, str]:
    """Load manifesto text files into a dictionary keyed by file stem."""
    text_dict: dict[str, str] = {}

    for root, _, files in os.walk(base_path):
        for file_name in files:
            if not file_name.endswith(".txt"):
                continue

            file_path = Path(root) / file_name
            file_id = file_path.stem

            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                text_dict[file_id] = file.read()

    return text_dict


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load metadata CSV."""
    return pd.read_csv(metadata_path, low_memory=False)


def merge_metadata_with_texts(
    metadata: pd.DataFrame,
    text_files_path: Path,
) -> pd.DataFrame:
    """Merge metadata with raw manifesto texts using document id."""
    unzip_archives(text_files_path)
    text_dict = load_text_files(text_files_path)

    merged = metadata.copy()
    merged["text"] = merged["id"].astype(str).map(text_dict)

    return merged[merged["text"].notna()].copy()
