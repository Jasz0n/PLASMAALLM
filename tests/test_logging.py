"""Tests for allm.core.logging."""

import json
import logging

from allm.core.logging import JsonFormatter, get_logger, setup_logging


def test_setup_is_idempotent() -> None:
    logger = setup_logging("INFO")
    setup_logging("INFO")
    assert len(logger.handlers) == 1


def test_level_applied() -> None:
    logger = setup_logging("debug")
    assert logger.level == logging.DEBUG


def test_child_logger_namespaced() -> None:
    assert get_logger("storage.sqlite").name == "allm.storage.sqlite"


def test_json_formatter_emits_valid_json() -> None:
    record = logging.LogRecord("allm.test", logging.INFO, __file__, 1, "hello %s", ("x",), None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello x"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "allm.test"
