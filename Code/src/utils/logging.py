"""Lightweight logging setup. One configurator, used by every entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(
    name: str = "regime_vol",
    *,
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Return a logger with stdout (and optional file) handlers.

    Idempotent: calling it twice with the same name reuses the existing
    handlers instead of stacking them, which keeps notebook reruns clean.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setFormatter(logging.Formatter(_DEFAULT_FMT))
        logger.addHandler(stream)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")) == log_file.resolve()
            for h in logger.handlers
        )
        if not already:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_DEFAULT_FMT))
            logger.addHandler(fh)

    return logger
