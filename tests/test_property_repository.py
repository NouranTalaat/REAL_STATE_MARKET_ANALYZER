from __future__ import annotations

from src.api.repositories.property_repository import (
    PropertyRepository,
)


class FakeResult:
    def __init__(self, rows=None, first=None):
        self._rows = rows or []
        self._first = first

    def mappings(self):
        return self

    def first(self):
        return self._first

    def all(self):
        return self._rows


class FakeConnection:
    def __init__(self, result):
        self.result = result

    def execute(self, query, parameters=None):
        self.query = query
        self.parameters = parameters
        return self.result


class FakeConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False


class FakeEngine:
    def __init__(self, result):
        self.result = result
        self.connection_instance = None

    def connect(self):
        self.connection_instance = FakeConnection(
            self.result
        )
        return FakeConnectionContext(
            self.connection_instance
        )


def test_count_properties():
    engine = FakeEngine(
        FakeResult(
            first={
                "total_properties": 39712,
            }
        )
    )

    repository = PropertyRepository(
        db_engine=engine
    )

    result = repository.count_properties()

    assert result == 39712


def test_count_properties_returns_zero_when_empty():
    engine = FakeEngine(
        FakeResult(first=None)
    )

    repository = PropertyRepository(
        db_engine=engine
    )

    result = repository.count_properties()

    assert result == 0


def test_search_properties_returns_rows():
    rows = [
        {
            "listing_id": 1,
            "property_type": "Apartment",
            "city": "Cairo",
        },
        {
            "listing_id": 2,
            "property_type": "Apartment",
            "city": "Cairo",
        },
    ]

    engine = FakeEngine(
        FakeResult(rows=rows)
    )

    repository = PropertyRepository(
        db_engine=engine
    )

    result = repository.search_properties(
        city="Cairo",
        property_type="Apartment",
        limit=10,
    )

    assert len(result) == 2
    assert result[0]["property_type"] == "Apartment"

    connection = engine.connection_instance

    assert connection.parameters["city"] == "Cairo"
    assert connection.parameters[
        "property_type"
    ] == "Apartment"
    assert connection.parameters["limit"] == 10