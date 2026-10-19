"""Retry-with-backoff helper for flaky LLM/tool calls.

The sleep function is injectable so tests exercise the real backoff logic
without spending wall-clock time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from app.llm.base import LLMPermanentError, LLMTransientError
from app.logging_config import get_logger

T = TypeVar("T")

logger = get_logger(__name__)

RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    LLMTransientError,
    httpx.TimeoutException,
    httpx.TransportError,
)

SleepFn = Callable[[float], Awaitable[None]]


class RetriesExhaustedError(LLMPermanentError):
    """All retry attempts failed; the caller should take its fallback path."""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        super().__init__(f"LLM call failed after {attempts} attempt(s): {last_error}")
        self.attempts = attempts
        self.last_error = last_error


async def call_with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    operation_name: str = "llm_call",
    sleep: SleepFn | None = None,
) -> T:
    """Run ``operation`` with exponential backoff on transient failures.

    Delays are ``base_delay * 2 ** (attempt - 1)``. Jitter is intentionally
    omitted to keep runs reproducible for the evaluation harness; for a
    high-concurrency deployment you would add it to avoid thundering herds.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    sleeper: SleepFn = sleep or asyncio.sleep
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "llm_call_retry",
                operation=operation_name,
                attempt=attempt,
                max_attempts=max_attempts,
                delay_seconds=round(delay, 3),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            await sleeper(delay)
        except LLMPermanentError as exc:
            logger.error(
                "llm_call_permanent_failure",
                operation=operation_name,
                attempt=attempt,
                error=str(exc),
            )
            raise

    if last_error is None:  # pragma: no cover - unreachable while max_attempts >= 1
        raise RuntimeError("retry loop exited without recording an error")
    logger.error(
        "llm_call_retries_exhausted",
        operation=operation_name,
        attempts=max_attempts,
        error=str(last_error),
    )
    raise RetriesExhaustedError(max_attempts, last_error)
