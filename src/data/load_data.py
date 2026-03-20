"""Utilities for loading prepared manifesto data from Parquet."""

from __future__ import annotations

import os

import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_manifestos_raw(url: str | None = None) -> pd.DataFrame:
    """Load the merged manifesto dataset from Parquet using DuckDB."""
    dataset_url = url or os.environ["URL_RAW"]

    query = f"""
        SELECT *
        FROM read_parquet('{dataset_url}')
    """

    con = duckdb.connect(database=":memory:")
    return con.execute(query).df()
