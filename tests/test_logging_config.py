"""Tests for logging configuration.

The laziness test is a regression guard: binding a logger at import time
freezes structlog's *default* configuration, so ``LOG_JSON`` silently had no
effect until ``get_logger`` stopped calling ``.bind()``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
import structlog

from app.logging_config import (
    bind_log_context,
    clear_log_context,
    configure_logging,
    get_logger,
    unbind_log_context,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    clear_log_context()
    yield
    clear_log_context()
    structlog.reset_defaults()


class TestConfigureLogging:
    def test_rejects_an_unknown_level(self) -> None:
        with pytest.raises(KeyError):
            configure_logging(level="CHATTY")

    def test_json_output_is_machine_readable(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json_output=True)
        get_logger("tests.example").info("something_happened", answer=42)

        line = capsys.readouterr().err.strip().splitlines()[-1]
        payload = json.loads(line)
        assert payload["event"] == "something_happened"
        assert payload["answer"] == 42
        assert payload["level"] == "info"
        assert payload["logger_name"] == "tests.example"
        assert "timestamp" in payload

    def test_console_output_is_not_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json_output=False)
        get_logger("tests.example").info("something_happened")

        err = capsys.readouterr().err
        assert "something_happened" in err
        with pytest.raises(json.JSONDecodeError):
            json.loads(err.strip().splitlines()[-1])

    def test_level_filtering_suppresses_debug(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="WARNING", json_output=True)
        logger = get_logger("tests.example")
        logger.debug("not_emitted")
        logger.warning("emitted")

        err = capsys.readouterr().err
        assert "not_emitted" not in err
        assert "emitted" in err


class TestLoggerLaziness:
    def test_module_level_loggers_pick_up_later_configuration(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A logger obtained *before* configure_logging must still honour it."""
        logger = get_logger("tests.early")  # as a module-level logger would be
        configure_logging(level="INFO", json_output=True)

        logger.info("late_configuration")

        payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert payload["event"] == "late_configuration"
        assert payload["logger_name"] == "tests.early"


class TestLogContext:
    def test_bound_context_appears_on_every_event(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json_output=True)
        bind_log_context(request_id="req-1", run_id="run-1")

        get_logger("tests.example").info("with_context")

        payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert payload["request_id"] == "req-1"
        assert payload["run_id"] == "run-1"

    def test_unbind_removes_a_single_key(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json_output=True)
        bind_log_context(request_id="req-1", run_id="run-1")
        unbind_log_context("run_id")

        get_logger("tests.example").info("partial_context")

        payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert payload["request_id"] == "req-1"
        assert "run_id" not in payload

    def test_clear_removes_everything(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", json_output=True)
        bind_log_context(request_id="req-1")
        clear_log_context()

        get_logger("tests.example").info("no_context")

        payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert "request_id" not in payload
