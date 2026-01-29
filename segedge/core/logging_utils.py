"""Logging setup utilities."""

from __future__ import annotations

import logging
import os


def setup_logging(log_path: str | None = None, level: int = logging.INFO) -> None:
    """Configure logging to file (and console) with a consistent format.

    Args:
        log_path (str | None): Optional log file path.
        level (int): Logging level.

    Examples:
        >>> callable(setup_logging)
        True
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
