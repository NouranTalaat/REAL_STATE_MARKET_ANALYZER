from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(
    app,
    raise_server_exceptions=False,
)


API_PREFIX = "/api/v2"


def valid_apartment_payload() -> dict:
    return {
        "market_type": "sale",
        "property_data": {
            "listing_type": "sale",
            "property_type": "Apartment",
            "completion_status": "Ready",
            "city": "Cairo",
            "town": "New Cairo",
            "district": "First Settlement",
            "subdistrict": "North",
            "furnished": "No",
            "listing_level": "Ground",
            "payment_method": "Cash",
            "lat": 30.0,
            "lon": 31.0,
            "bedrooms": 3,
            "bathrooms": 2,
            "area_value": 150,
            "is_premium": 0,
            "is_featured": 0,
            "images_count": 10,
            "has_view_360": 0,
            "has_video": 1,
            "has_360_view": 0,
            "has_amenities": 1,
            "has_district": 1,
            "has_subdistrict": 1,
            "is_new_construction": 1,
            "is_direct_from_dev": 0,
            "is_exclusive": 0,
            "is_verified": 1,
            "agent_is_super": 1,
            "agent_languages": "English,Arabic",
            "listed_date": "2026-01-15",
        },
    }


def valid_chalet_payload() -> dict:
    payload = valid_apartment_payload()

    payload["property_data"].update(
        {
            "property_type": "Chalet",
            "city": "North Coast",
            "town": "Marassi",
            "district": "North Coast",
            "subdistrict": "Marassi",
            "lat": 30.9,
            "lon": 28.9,
            "area_value": 180,
        }
    )

    return payload


# ============================================================
# SYSTEM ENDPOINTS
# ============================================================


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["api_version"] == "v2"
    assert data["version"] == "2.0.0"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["api_version"] == "v2"
    assert data["version"] == "2.0.0"


def test_readiness():
    response = client.get("/ready")

    assert response.status_code in (200, 503)

    data = response.json()

    assert data["status"] in (
        "ready",
        "not_ready",
    )

    assert data["api_version"] == "v2"

    assert "checks" in data
    assert "database" in data["checks"]
    assert "models" in data["checks"]


# ============================================================
# REQUEST CONTEXT / CORRELATION
# ============================================================


def test_request_id_is_generated():
    response = client.get("/health")

    assert response.status_code == 200

    request_id = response.headers.get("X-Request-ID")

    assert request_id is not None
    assert request_id != ""
    assert len(request_id) <= 128


def test_request_id_is_preserved():
    request_id = "integration-test-request-001"

    response = client.get(
        "/health",
        headers={
            "X-Request-ID": request_id,
        },
    )

    assert response.status_code == 200

    assert (
        response.headers["X-Request-ID"]
        == request_id
    )


def test_request_duration_header_exists():
    response = client.get("/health")

    assert response.status_code == 200

    duration = response.headers.get(
        "X-Request-Duration-Ms"
    )

    assert duration is not None

    assert float(duration) >= 0


# ============================================================
# API V2 ROUTING
# ============================================================


def test_prediction_status_v2():
    response = client.get(
        f"{API_PREFIX}/predictions/status"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["error"] is None

    assert "data" in body
    assert "meta" in body

    assert "request_id" in body["meta"]
    assert "timestamp" in body["meta"]
    assert body["meta"]["version"] == "2.0.0"


def test_old_prediction_route_is_not_exposed():
    response = client.get(
        "/predictions/status"
    )

    assert response.status_code == 404


# ============================================================
# PREDICTION TESTS
# ============================================================


def test_valuate_valid_apartment():
    response = client.post(
        f"{API_PREFIX}/predictions/valuate",
        json=valid_apartment_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    # Standard API response contract
    assert body["success"] is True
    assert body["error"] is None

    assert "meta" in body
    assert "request_id" in body["meta"]
    assert "timestamp" in body["meta"]
    assert body["meta"]["version"] == "2.0.0"

    # Business response
    data = body["data"]

    assert data["market_type"] == "sale"

    assert (
        data["market_value"]
        == "Residential for Sale"
    )

    assert (
        data["property_type"]
        == "Apartment"
    )

    assert isinstance(
        data["estimated_value_egp"],
        float,
    )

    assert (
        data["estimated_value_egp"]
        >= 0
    )

    assert "confidence" in data
    assert "model_predictions" in data

    assert len(
        data["model_predictions"]
    ) == 3


def test_valuate_rejects_target_leakage():
    payload = valid_apartment_payload()

    payload["property_data"]["price_egp"] = 2_500_000

    response = client.post(
        f"{API_PREFIX}/predictions/valuate",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    # Standard error response contract
    assert data["success"] is False
    assert data["data"] is None
    assert data["error"] is not None

    assert (
        data["error"]["code"]
        == "PREDICTION_ERROR"
    )

    assert (
        "price_egp"
        in data["error"]["message"]
    )

    assert "meta" in data
    assert data["meta"]["request_id"]
    assert (
        data["meta"]["version"]
        == "2.0.0"
    )

    # The old FastAPI "detail" contract
    # must NOT exist anymore.
    assert "detail" not in data


def test_valuate_rejects_invalid_market():
    payload = valid_apartment_payload()

    payload["market_type"] = "something"

    response = client.post(
        f"{API_PREFIX}/predictions/valuate",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    # Standard error response contract
    assert data["success"] is False
    assert data["data"] is None
    assert data["error"] is not None

    assert (
        data["error"]["code"]
        == "PREDICTION_ERROR"
    )

    assert (
        "sale"
        in data["error"]["message"]
    )

    assert (
        "rent"
        in data["error"]["message"]
    )

    assert "meta" in data
    assert data["meta"]["request_id"]
    assert (
        data["meta"]["version"]
        == "2.0.0"
    )

    assert "detail" not in data


def test_valuate_rejects_negative_area():
    payload = valid_apartment_payload()

    payload["property_data"]["area_value"] = -100

    response = client.post(
        f"{API_PREFIX}/predictions/valuate",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    # Standard error response contract
    assert data["success"] is False
    assert data["data"] is None
    assert data["error"] is not None

    assert (
        data["error"]["code"]
        == "PREDICTION_ERROR"
    )

    assert (
        "area_value"
        in data["error"]["message"]
    )

    assert (
        "negative"
        in data["error"]["message"]
    )

    assert "meta" in data
    assert data["meta"]["request_id"]
    assert (
        data["meta"]["version"]
        == "2.0.0"
    )

    assert "detail" not in data


def test_valuate_uses_specialized_chalet_routing():
    response = client.post(
        f"{API_PREFIX}/predictions/valuate",
        json=valid_chalet_payload(),
    )

    assert response.status_code == 200

    body = response.json()

    # Standard API response contract
    assert body["success"] is True
    assert body["error"] is None

    assert "meta" in body
    assert "request_id" in body["meta"]
    assert "timestamp" in body["meta"]
    assert body["meta"]["version"] == "2.0.0"

    # Business response
    data = body["data"]

    assert data["market_type"] == "sale"

    assert (
        data["property_type"]
        == "Chalet"
    )

    assert (
        data["model_resolution"]
        == "specialized"
    )

    assert (
        data["estimated_value_egp"]
        >= 0
    )

    assert len(
        data["model_predictions"]
    ) == 3