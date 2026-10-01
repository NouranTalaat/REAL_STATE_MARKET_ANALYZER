import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.database import engine
from src.ml.train import train_all


def load_training_data() -> pd.DataFrame:
    query = """
        SELECT *
        FROM analytics.property_listings
        WHERE price_egp IS NOT NULL
          AND price_egp > 0
          AND offering_type IS NOT NULL;
    """

    return pd.read_sql(query, engine)


def main() -> None:
    print("=" * 70)
    print("REAL ESTATE MARKET INTELLIGENCE PLATFORM")
    print("PHASE 7 - ML TRAINING")
    print("=" * 70)

    print("\n[1/3] Loading training data from SQL Server...")

    df = load_training_data()

    print(f"Loaded {len(df):,} records.")

    print("\nMarket distribution:")
    print(
        df["offering_type"]
        .value_counts()
        .to_string()
    )

    print("\n[2/3] Training Sale and Rent models...")

    results = train_all(df)

    print("\n[3/3] Training completed successfully.")

    print("\nModel Results")
    print("-" * 70)

    for result in results:
        print(f"\nMarket: {result['market_type']}")
        print(f"Market value: {result['market_value']}")
        print(f"Training rows: {result['training_rows']:,}")
        print(f"Test rows: {result['test_rows']:,}")

        print("\nMetrics:")

        for metric, value in result["metrics"].items():
            print(f"  {metric}: {value:,.4f}")

        print(f"\nModel: {result['model_path']}")
        print(f"Metadata: {result['metadata_path']}")

    print("\n" + "=" * 70)
    print("PHASE 7 ML TRAINING FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()