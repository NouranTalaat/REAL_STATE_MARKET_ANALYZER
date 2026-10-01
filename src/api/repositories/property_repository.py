from __future__ import annotations

from typing import Any

from sqlalchemy import text

from src.database import engine


class PropertyRepository:
    """
    Repository responsible for property-related database access.

    This layer owns SQL queries and database interaction.
    API routes and services should not access SQLAlchemy
    directly.
    """

    def __init__(self, db_engine=engine) -> None:
        self.engine = db_engine

    def count_properties(self) -> int:
        query = text(
            """
            SELECT COUNT(*) AS total_properties
            FROM analytics.property_listings;
            """
        )

        with self.engine.connect() as connection:
            result = connection.execute(query).mappings().first()

        if result is None:
            return 0

        return int(result["total_properties"])

    def search_properties(
        self,
        *,
        city: str | None = None,
        property_type: str | None = None,
        listing_type: str | None = None,
        offering_type: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []

        parameters: dict[str, Any] = {
            "limit": limit,
        }

        if city:
            conditions.append("city = :city")
            parameters["city"] = city

        if property_type:
            conditions.append(
                "property_type = :property_type"
            )
            parameters["property_type"] = property_type

        if listing_type:
            conditions.append(
                "listing_type = :listing_type"
            )
            parameters["listing_type"] = listing_type

        if offering_type:
            conditions.append(
                "offering_type = :offering_type"
            )
            parameters["offering_type"] = offering_type

        where_clause = ""

        if conditions:
            where_clause = (
                "WHERE " + " AND ".join(conditions)
            )

        query = text(
            f"""
            SELECT TOP (:limit)
                listing_id,
                category,
                listing_type,
                property_type,
                offering_type,
                price_egp,
                price_period,
                city,
                town,
                district,
                bedrooms,
                bathrooms,
                area_value,
                area_unit,
                furnished,
                listing_level,
                is_premium,
                is_verified,
                listed_date,
                price_per_sqm
            FROM analytics.property_listings
            {where_clause}
            ORDER BY listed_date DESC;
            """
        )

        with self.engine.connect() as connection:
            result = connection.execute(
                query,
                parameters,
            )

            return [
                dict(row)
                for row in result.mappings().all()
            ]