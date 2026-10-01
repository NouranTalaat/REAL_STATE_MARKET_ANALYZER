from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


# ============================================================
# SECURITY HEADERS
# ============================================================


def test_security_headers_are_present():
    response = client.get("/health")

    assert response.status_code == 200

    assert (
        response.headers["x-content-type-options"]
        == "nosniff"
    )

    assert (
        response.headers["x-frame-options"]
        == "DENY"
    )

    assert (
        response.headers["referrer-policy"]
        == "no-referrer"
    )

    assert (
        response.headers["permissions-policy"]
        == (
            "camera=(), microphone=(), "
            "geolocation=()"
        )
    )


# ============================================================
# REQUEST ID
# ============================================================


def test_request_id_is_generated():
    response = client.get("/health")

    assert response.status_code == 200

    request_id = response.headers.get(
        "x-request-id"
    )

    assert request_id
    assert len(request_id) > 0


def test_request_id_is_preserved():
    request_id = "security-test-request-001"

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": request_id,
        },
    )

    assert response.status_code == 200

    assert (
        response.headers["x-request-id"]
        == request_id
    )


# ============================================================
# REQUEST DURATION
# ============================================================


def test_request_duration_header_is_present():
    response = client.get("/health")

    assert response.status_code == 200

    duration = response.headers.get(
        "x-request-duration-ms"
    )

    assert duration is not None

    duration_value = float(duration)

    assert duration_value >= 0


# ============================================================
# TRUSTED HOST
# ============================================================


def test_trusted_host_accepts_localhost():
    response = client.get(
        "/health",
        headers={
            "Host": "localhost",
        },
    )

    assert response.status_code == 200


def test_trusted_host_rejects_unknown_host():
    response = client.get(
        "/health",
        headers={
            "Host": "malicious.example",
        },
    )

    assert response.status_code == 400


# ============================================================
# REQUEST SIZE LIMIT
# ============================================================


def test_request_size_limit_rejects_large_payload():
    large_payload = b"x" * (
        11 * 1024 * 1024
    )

    response = client.post(
        "/api/v2/predictions/valuate",
        content=large_payload,
        headers={
            "Content-Type": "application/json",
        },
    )

    assert response.status_code == 413

    body = response.json()

    assert body["success"] is False
    assert body["error"]["code"] == (
        "REQUEST_TOO_LARGE"
    )


# ============================================================
# CORS
# ============================================================


def test_cors_allows_configured_origin():
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200

    assert (
        response.headers.get(
            "access-control-allow-origin"
        )
        == "http://localhost"
    )


def test_cors_does_not_allow_unknown_origin():
    response = client.options(
        "/health",
        headers={
            "Origin": "http://malicious.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert (
        response.headers.get(
            "access-control-allow-origin"
        )
        != "http://malicious.example"
    )