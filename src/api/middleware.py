from __future__ import annotations

import time
import uuid

from starlette.types import (
    ASGIApp,
    Message,
    Receive,
    Scope,
    Send,
)

from src.core.config import get_settings
from src.utils.logging import get_logger, log_structured


REQUEST_ID_HEADER = "x-request-id"
REQUEST_DURATION_HEADER = "x-request-duration-ms"

MAX_REQUEST_ID_LENGTH = 128

logger = get_logger(__name__)
settings = get_settings()


def _get_request_id(scope: Scope) -> str:
    """
    Return a valid incoming request ID or generate a new UUID.
    """

    headers = dict(scope.get("headers", []))

    incoming_request_id = headers.get(
        REQUEST_ID_HEADER.encode("latin-1")
    )

    if incoming_request_id:
        request_id = incoming_request_id.decode(
            "latin-1",
            errors="ignore",
        ).strip()

        if (
            request_id
            and len(request_id) <= MAX_REQUEST_ID_LENGTH
        ):
            return request_id

    return str(uuid.uuid4())


class RequestContextMiddleware:
    """
    Production request-context middleware.

    Responsibilities:
    - Generate or preserve X-Request-ID.
    - Store request_id in request.state.
    - Measure request execution time.
    - Return request metadata through response headers.
    - Emit structured request logs.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:

        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _get_request_id(scope)
        start_time = time.perf_counter()

        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id

        method = scope.get(
            "method",
            "UNKNOWN",
        )

        path = scope.get(
            "path",
            "UNKNOWN",
        )

        async def send_with_context(
            message: Message,
        ) -> None:

            if message["type"] == "http.response.start":
                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                status_code = message.get(
                    "status",
                    500,
                )

                headers = list(
                    message.get("headers", [])
                )

                headers.append(
                    (
                        REQUEST_ID_HEADER.encode(
                            "latin-1"
                        ),
                        request_id.encode(
                            "latin-1"
                        ),
                    )
                )

                headers.append(
                    (
                        REQUEST_DURATION_HEADER.encode(
                            "latin-1"
                        ),
                        f"{duration_ms:.2f}".encode(
                            "latin-1"
                        ),
                    )
                )

                message["headers"] = headers

                log_structured(
                    logger,
                    20,
                    "HTTP request completed",
                    service=settings.service_name,
                    environment=settings.environment,
                    request_id=request_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=round(
                        duration_ms,
                        2,
                    ),
                )

            await send(message)

        await self.app(
            scope,
            receive,
            send_with_context,
        )