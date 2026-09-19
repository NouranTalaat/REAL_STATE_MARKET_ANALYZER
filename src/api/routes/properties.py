from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from src.database import engine


router = APIRouter(
    prefix="/properties",
    tags=["Properties"],
)


@router.get("/count")
def property_count():

    query = text(
        """
        SELECT COUNT(*) AS total_properties
        FROM analytics.property_listings;
        """
    )

    try:
        with engine.connect() as connection:
            result = connection.execute(query).mappings().first()

        return {
            "total_properties": result["total_properties"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load property count: {str(e)}",
        )


@router.get("/search")
def search_properties(
    city: str | None = Query(default=None),
    property_type: str | None = Query(default=None),
    listing_type: str | None = Query(default=None),
    offering_type: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):

    conditions = []

    parameters = {
        "limit": limit
    }

    if city:
        conditions.append("city = :city")
        parameters["city"] = city

    if property_type:
        conditions.append("property_type = :property_type")
        parameters["property_type"] = property_type

    if listing_type:
        conditions.append("listing_type = :listing_type")
        parameters["listing_type"] = listing_type

    if offering_type:
        conditions.append("offering_type = :offering_type")
        parameters["offering_type"] = offering_type

    where_clause = ""

    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

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

    try:

        with engine.connect() as connection:
            result = connection.execute(
                query,
                parameters
            )

            rows = [
                dict(row)
                for row in result.mappings().all()
            ]

        return {
            "count": len(rows),
            "data": rows,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search properties: {str(e)}",
        )