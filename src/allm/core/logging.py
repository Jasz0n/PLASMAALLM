"""Logging setup.

Design decisions
----------------
- Standard-library ``logging`` only: every third-party tool integrates
  with it, and it keeps the core dependency-free.
- All ALLM loggers live under the ``"allm"`` namespace so applications
  embedding ALLM can control our verbosity independently.
- Optional JSON output so experiment logs are machine-parseable.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

ROOT_LOGGER_NAME = "allm"


class JsonFormatter(logging.Formatter):
    """Format records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_format: bool = False) -> logging.Logger:
    """Configure and return the ``allm`` root logger.

    Idempotent: calling it again reconfigures rather than duplicating
    handlers, so tests and notebooks can call it freely.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, e.g. ``get_logger("storage.sqlite")``."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
