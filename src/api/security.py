from __future__ import annotations

from collections.abc import Awaitable, Callable

from starlette.types import (
    ASGIApp,
    Message,
    Receive,
    Scope,
    Send,
)


# ============================================================
# SECURITY HEADERS
# ============================================================

class SecurityHeadersMiddleware:
    """
    Add baseline HTTP security headers to every HTTP response.

    The middleware intentionally avoids overly restrictive
    browser policies that could break API documentation,
    development tooling, or future frontend integration.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        environment: str = "development",
    ) -> None:
        self.app = app
        self.environment = environment.lower().strip()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:

        if scope["type"] != "http":
            await self.app(
                scope,
                receive,
                send,
            )
            return

        async def send_with_security_headers(
            message: Message,
        ) -> None:

            if message["type"] == "http.response.start":
                headers = list(
                    message.get("headers", [])
                )

                security_headers = [
                    (
                        b"x-content-type-options",
                        b"nosniff",
                    ),
                    (
                        b"x-frame-options",
                        b"DENY",
                    ),
                    (
                        b"referrer-policy",
                        b"no-referrer",
                    ),
                    (
                        b"permissions-policy",
                        b"camera=(), microphone=(), geolocation=()",
                    ),
                ]

                # HSTS should only be enabled when the API is
                # actually served through HTTPS.
                if self.environment == "production":
                    security_headers.append(
                        (
                            b"strict-transport-security",
                            b"max-age=31536000; includeSubDomains",
                        )
                    )

                existing_header_names = {
                    name.lower()
                    for name, _ in headers
                }

                for name, value in security_headers:
                    if name not in existing_header_names:
                        headers.append(
                            (name, value)
                        )

                message["headers"] = headers

            await send(message)

        await self.app(
            scope,
            receive,
            send_with_security_headers,
        )


# ============================================================
# REQUEST SIZE PROTECTION
# ============================================================

class RequestSizeLimitMiddleware:
    """
    Reject HTTP requests whose Content-Length exceeds
    the configured maximum request size.

    This is a lightweight baseline protection against
    accidentally or intentionally sending excessively
    large API payloads.

    Note:
    Requests using chunked transfer encoding without a
    Content-Length header are not rejected here because
    determining their final size requires buffering the
    complete request body.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_request_size_bytes: int,
    ) -> None:
        self.app = app
        self.max_request_size_bytes = (
            max_request_size_bytes
        )

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:

        if scope["type"] != "http":
            await self.app(
                scope,
                receive,
                send,
            )
            return

        headers = dict(
            scope.get("headers", [])
        )

        content_length = headers.get(
            b"content-length"
        )

        if content_length is not None:
            try:
                request_size = int(
                    content_length
                )
            except ValueError:
                request_size = 0

            if (
                request_size
                > self.max_request_size_bytes
            ):
                response = (
                    b'{"success":false,'
                    b'"data":null,'
                    b'"error":{'
                    b'"code":"REQUEST_TOO_LARGE",'
                    b'"message":"Request body exceeds the maximum allowed size.",'
                    b'"details":null'
                    b'}}'
                )

                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [
                            (
                                b"content-type",
                                b"application/json",
                            ),
                            (
                                b"content-length",
                                str(
                                    len(response)
                                ).encode(
                                    "latin-1"
                                ),
                            ),
                        ],
                    }
                )

                await send(
                    {
                        "type": "http.response.body",
                        "body": response,
                    }
                )

                return

        await self.app(
            scope,
            receive,
            send,
        )