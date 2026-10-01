from __future__ import annotations

from src.api.services.property_service import (
    PropertyService,
)


class FakePropertyRepository:
    def count_properties(self):
        return 39712

    def search_properties(
        self,
        *,
        city=None,
        property_type=None,
        listing_type=None,
        offering_type=None,
        limit=20,
    ):
        return [
            {
                "listing_id": 1,
                "city": city,
                "property_type": property_type,
            }
        ]


def test_property_service_count():
    service = PropertyService(
        repository=FakePropertyRepository()
    )

    result = service.get_property_count()

    assert result == 39712


def test_property_service_search():
    service = PropertyService(
        repository=FakePropertyRepository()
    )

    result = service.search_properties(
        city="Cairo",
        property_type="Apartment",
        limit=10,
    )

    assert len(result) == 1
    assert result[0]["city"] == "Cairo"
    assert result[0]["property_type"] == "Apartment"