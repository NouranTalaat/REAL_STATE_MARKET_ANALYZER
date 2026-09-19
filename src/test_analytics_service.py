from analytics_service import (
    get_market_summary,
    get_robust_metrics,
    get_location_intelligence,
    get_opportunity_signals,
    get_data_quality_status,
)


def main():
    market_summary = get_market_summary()
    robust_metrics = get_robust_metrics()
    location_intelligence = get_location_intelligence()
    opportunity_signals = get_opportunity_signals()
    data_quality = get_data_quality_status()

    print("\n=== MARKET SUMMARY ===")
    print(market_summary.head())

    print("\n=== ROBUST METRICS ===")
    print(robust_metrics.head())

    print("\n=== LOCATION INTELLIGENCE ===")
    print(location_intelligence.head())

    print("\n=== OPPORTUNITY SIGNALS ===")
    print(opportunity_signals.head())

    print("\n=== DATA QUALITY ===")
    print(data_quality)


if __name__ == "__main__":
    main()