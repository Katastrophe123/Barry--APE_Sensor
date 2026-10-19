"""Request-scoped correlation id and access logging."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from time import perf_counter

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.api.errors import REQUEST_ID_HEADER
from app.logging_config import bind_log_context, clear_log_context, get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, binds it to the log context and logs the result.

    An inbound ``X-Request-ID`` is honoured so ids correlate across services.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = request_id
        clear_log_context()
        bind_log_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_errored",
                duration_ms=int((perf_counter() - started) * 1000),
            )
            clear_log_context()
            raise

        duration_ms = int((perf_counter() - started) * 1000)
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        clear_log_context()
        return response
