from __future__ import annotations

from typing import Any

from src.api.repositories.property_repository import (
    PropertyRepository,
)


class PropertyService:
    """
    Application service for property-related operations.

    The service coordinates application-level behavior
    while the repository owns database access.
    """

    def __init__(
        self,
        repository: PropertyRepository | None = None,
    ) -> None:
        self.repository = (
            repository
            if repository is not None
            else PropertyRepository()
        )

    def get_property_count(self) -> int:
        return self.repository.count_properties()

    def search_properties(
        self,
        *,
        city: str | None = None,
        property_type: str | None = None,
        listing_type: str | None = None,
        offering_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self.repository.search_properties(
            city=city,
            property_type=property_type,
            listing_type=listing_type,
            offering_type=offering_type,
            limit=limit,
        )