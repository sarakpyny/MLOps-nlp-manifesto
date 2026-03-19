"""Logging configuration utilities."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str = "logs", log_file: str = "pipeline.log") -> logging.Logger:
    """Configure root logging to console and file.

    Parameters
    ----------
    log_dir : str
        Directory where logs are stored.
    log_file : str
        Log filename.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
