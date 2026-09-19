from sql_analytics import (
    load_market_summary,
    load_robust_metrics,
    load_location_intelligence,
    load_opportunity_signals,
    load_data_quality_monitor,
)


def main():
    market_summary = load_market_summary()
    robust_metrics = load_robust_metrics()
    location_intelligence = load_location_intelligence()
    opportunity_signals = load_opportunity_signals()
    data_quality = load_data_quality_monitor()

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