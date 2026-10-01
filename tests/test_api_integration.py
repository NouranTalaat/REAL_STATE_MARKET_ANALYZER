from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


# ============================================================
# SYSTEM INTEGRATION
# ============================================================


def test_home_endpoint_integration():
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["api_version"] == "v2"
    assert "service" in body
    assert "version" in body


def test_health_endpoint_integration():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["api_version"] == "v2"


# ============================================================
# OPENAPI INTEGRATION
# ============================================================


def test_openapi_endpoint_integration():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema

    assert (
        schema["info"]["version"]
        == "2.0.0"
    )


def test_docs_endpoint_integration():
    response = client.get("/docs")

    assert response.status_code == 200

    assert "swagger" in response.text.lower()


def test_redoc_endpoint_integration():
    response = client.get("/redoc")

    assert response.status_code == 200

    assert "redoc" in response.text.lower()


# ============================================================
# API VERSIONING
# ============================================================


def test_v2_prediction_status_route_exists():
    response = client.get(
        "/api/v2/predictions/status"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["data"] is not None
    assert body["meta"]["version"] == "2.0.0"


def test_old_prediction_route_is_not_available():
    response = client.get(
        "/predictions/status"
    )

    assert response.status_code == 404


# ============================================================
# STANDARD RESPONSE CONTRACT
# ============================================================


def test_prediction_status_response_contract():
    response = client.get(
        "/api/v2/predictions/status"
    )

    assert response.status_code == 200

    body = response.json()

    assert set(body.keys()) == {
        "success",
        "data",
        "error",
        "meta",
    }

    assert body["success"] is True
    assert body["error"] is None

    assert (
        "request_id"
        in body["meta"]
    )

    assert (
        "version"
        in body["meta"]
    )


# ============================================================
# VALIDATION INTEGRATION
# ============================================================


def test_prediction_validation_error_contract():
    response = client.post(
        "/api/v2/predictions/valuate",
        json={
            "market_type": "invalid-market",
            "property_data": {
                "property_type": "Apartment",
            },
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["success"] is False
    assert body["data"] is None
    assert body["error"] is not None

    assert (
        body["error"]["code"]
        == "PREDICTION_ERROR"
    )


def test_prediction_extra_field_is_rejected():
    response = client.post(
        "/api/v2/predictions/valuate",
        json={
            "market_type": "sale",
            "property_data": {
                "property_type": "Apartment",
            },
            "unexpected_field": "should-fail",
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["success"] is False
    assert body["data"] is None
    assert body["error"] is not None


# ============================================================
# API ROUTING INTEGRATION
# ============================================================


def test_unknown_api_route_returns_404():
    response = client.get(
        "/api/v2/this-route-does-not-exist"
    )

    assert response.status_code == 404


def test_unknown_method_returns_405():
    response = client.patch(
        "/health"
    )

    assert response.status_code == 405