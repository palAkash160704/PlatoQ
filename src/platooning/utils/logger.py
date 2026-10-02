"""
Centralised logging configuration.

Provides a consistent log format across all modules:

    2026-10-02 11:30:00 | INFO | simulation.simulator | Starting simulator

Usage::

    from platooning.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Hello from %s", __name__)
"""

from __future__ import annotations

import logging
import sys

# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a console handler.

    This function is idempotent — calling it multiple times will not add
    duplicate handlers.

    Parameters
    ----------
    level : str
        Logging level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
    """
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    """
    return logging.getLogger(name)
