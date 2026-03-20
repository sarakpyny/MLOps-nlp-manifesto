"""Tests for Parquet data loading."""

import pandas as pd

from src.data.load_data import load_manifestos_raw


def test_load_manifestos_raw_uses_explicit_url(monkeypatch):
    """Loader should read from the provided Parquet URL."""
    expected = pd.DataFrame({"text": ["hello"], "label": ["A"]})
    captured = {}

    class FakeResult:
        def df(self):
            return expected

    class FakeConn:
        def execute(self, query):
            captured["query"] = query
            return FakeResult()

    import src.data.load_data as mod

    monkeypatch.setattr(mod.duckdb, "connect",
                        lambda database=":memory:": FakeConn())

    result = load_manifestos_raw("dummy.parquet")

    assert result.equals(expected)
    assert "dummy.parquet" in captured["query"]


def test_load_manifestos_raw_uses_env_url(monkeypatch):
    """Loader should fall back to URL_RAW from the environment."""
    expected = pd.DataFrame({"text": ["bonjour"]})
    captured = {}

    class FakeResult:
        def df(self):
            return expected

    class FakeConn:
        def execute(self, query):
            captured["query"] = query
            return FakeResult()

    import src.data.load_data as mod

    monkeypatch.setattr(mod.duckdb, "connect",
                        lambda database=":memory:": FakeConn())
    monkeypatch.setenv("URL_RAW", "env_dataset.parquet")

    result = load_manifestos_raw()

    assert result.equals(expected)
    assert "env_dataset.parquet" in captured["query"]
