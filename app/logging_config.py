"""Structured logging setup.

Uses ``structlog`` with context variables so that ``request_id`` / ``run_id``
are attached to every log line emitted downstream of a bound context, without
having to thread a logger through every function signature.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars

__all__ = [
    "bind_log_context",
    "clear_log_context",
    "configure_logging",
    "get_logger",
    "unbind_log_context",
]


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Configure stdlib logging and structlog to share one renderer."""
    # Native structlog processors only: the logger factory below is structlog's
    # own PrintLogger, so the ``structlog.stdlib.*`` processors (which expect a
    # ``logging.Logger``) must not be used here.
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Route uvicorn / sqlalchemy records through the same stream & level.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=logging.getLevelNamesMapping()[level.upper()],
        force=True,
    )
    for noisy in ("uvicorn.access", "uvicorn.error", "sqlalchemy.engine"):
        logging.getLogger(noisy).propagate = True


def get_logger(name: str | None = None) -> Any:
    """Return a lazily-bound logger, namespaced by ``name`` when given.

    Initial values are passed to ``get_logger`` rather than via ``.bind()``:
    ``bind()`` materialises the logger immediately against whatever config is
    active, which at module-import time is structlog's default. Passing them
    here keeps the proxy lazy so ``configure_logging`` is always applied.
    """
    # ``logger_name`` (not ``logger``, which collides with wrap_logger's own
    # parameter) is the key ConsoleRenderer/JSONRenderer show as the namespace.
    return structlog.get_logger(logger_name=name) if name else structlog.get_logger()


def bind_log_context(**kwargs: Any) -> None:
    bind_contextvars(**kwargs)


def unbind_log_context(*keys: str) -> None:
    unbind_contextvars(*keys)


def clear_log_context() -> None:
    clear_contextvars()
