"""Tests for logging configuration."""

from pathlib import Path

from src.utils.logging_config import setup_logging


def test_setup_logging_creates_log_file(tmp_path: Path):
    """Logging setup should create the log directory and log file."""
    logger = setup_logging(log_dir=str(tmp_path), log_file="test.log")
    logger.info("test message")

    log_file = tmp_path / "test.log"

    assert log_file.exists()
    assert "test message" in log_file.read_text(encoding="utf-8")
