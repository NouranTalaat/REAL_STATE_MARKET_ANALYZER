from __future__ import annotations

from typing import Any

from src.api.schemas.common import APIResponse, ErrorDetail, ResponseMeta


def build_error_response(
    *,
    code: str,
    message: str,
    request_id: str,
    version: str,
    details: dict[str, Any] | list[Any] | None = None,
) -> APIResponse[None]:
    """
    Build a standardized API error response.
    """

    return APIResponse(
        success=False,
        data=None,
        meta=ResponseMeta(
            request_id=request_id,
            version=version,
        ),
        error=ErrorDetail(
            code=code,
            message=message,
            details=details,
        ),
    )