
"""
PHASE 7 FINAL SANITY TEST
Real Estate Valuation Engine

Tests:
- Qualified specialized routing
- Market fallback routing
- Sale and Rent
- Strict input validation compatibility
- No target/leakage fields sent to inference
"""

import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ENGINE IMPORT
# ============================================================

from src.ml.valuation_engine import RealEstateValuationEngine


# ============================================================
# DATASET
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "propertyfinder.csv"
)


# ============================================================
# LEAKAGE / TARGET FIELDS
# These fields must NEVER be sent to prediction.
# ============================================================

FORBIDDEN_INPUT_FIELDS = {
    "price_egp",
    "price_per_sqm",
    "target",
    "log_price",
    "log_target",
}


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    {
        "market": "sale",
        "property_type": "Chalet",
        "expected": "specialized",
    },
    {
        "market": "sale",
        "property_type": "Townhouse",
        "expected": "specialized",
    },
    {
        "market": "sale",
        "property_type": "Twin House",
        "expected": "specialized",
    },
    {
        "market": "sale",
        "property_type": "Apartment",
        "expected": "market",
    },
    {
        "market": "rent",
        "property_type": "Duplex",
        "expected": "specialized",
    },
    {
        "market": "rent",
        "property_type": "Penthouse",
        "expected": "specialized",
    },
    {
        "market": "rent",
        "property_type": "Twin House",
        "expected": "market",
    },
    {
        "market": "rent",
        "property_type": "Apartment",
        "expected": "market",
    },
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )


# ============================================================
# FIND REAL SAMPLE
# ============================================================

def find_sample(
    df: pd.DataFrame,
    market: str,
    property_type: str,
) -> dict:

    offering_type = (
        "Residential for Sale"
        if market == "sale"
        else "Residential for Rent"
    )

    mask = (
        (df["offering_type"] == offering_type)
        &
        (
            df["property_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            == property_type.lower()
        )
    )

    matches = df.loc[mask]

    if matches.empty:
        raise ValueError(
            f"No sample found for "
            f"{market} / {property_type}"
        )

    raw_sample = matches.iloc[0].to_dict()

    # --------------------------------------------------------
    # Remove target / leakage fields.
    #
    # The real dataset contains the actual property price,
    # but inference must behave as if the price is unknown.
    # --------------------------------------------------------

    clean_sample = {
        key: value
        for key, value in raw_sample.items()
        if key not in FORBIDDEN_INPUT_FIELDS
    }

    return clean_sample


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 80)
    print("PHASE 7 FINAL VALUATION ENGINE SANITY TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    print(
        f"Dataset loaded: {len(df):,} rows"
    )

    # --------------------------------------------------------
    # Load engine
    # --------------------------------------------------------

    engine = RealEstateValuationEngine()

    passed = 0
    failed = 0

    # ========================================================
    # ROUTING TESTS
    # ========================================================

    for case in TEST_CASES:

        market = case["market"]
        property_type = case["property_type"]
        expected = case["expected"]

        print()
        print("-" * 80)

        print(
            f"Testing: "
            f"{market.upper()} | "
            f"{property_type}"
        )

        try:

            # ------------------------------------------------
            # Get real property sample
            # ------------------------------------------------

            sample = find_sample(
                df=df,
                market=market,
                property_type=property_type,
            )

            # ------------------------------------------------
            # Safety check:
            # Ensure forbidden fields are NOT present.
            # ------------------------------------------------

            forbidden_present = (
                set(sample.keys())
                & FORBIDDEN_INPUT_FIELDS
            )

            if forbidden_present:

                raise AssertionError(
                    "Test sample still contains "
                    f"forbidden fields: "
                    f"{sorted(forbidden_present)}"
                )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            result = engine.predict(
                property_data=sample,
                market_type=market,
            )

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            actual = result["model_resolution"]

            predicted_value = (
                result["estimated_value_egp"]
            )

            confidence = result["confidence"]

            print(
                f"Expected routing : {expected}"
            )

            print(
                f"Actual routing   : {actual}"
            )

            print(
                f"Estimated value  : "
                f"{predicted_value:,.2f} EGP"
            )

            print(
                f"Confidence       : "
                f"{confidence['level']} "
                f"({confidence['score']})"
            )

            # ------------------------------------------------
            # Validate routing
            # ------------------------------------------------

            if actual != expected:

                print("❌ FAILED")

                failed += 1

                continue

            print("✅ PASSED")

            passed += 1

        except Exception as exc:

            print("❌ FAILED")

            print(
                f"Error: {type(exc).__name__}: {exc}"
            )

            failed += 1

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        f"Passed: {passed}/{len(TEST_CASES)}"
    )

    print(
        f"Failed: {failed}/{len(TEST_CASES)}"
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if failed == 0:

        print()
        print(
            "🎉 PHASE 7 SANITY TEST PASSED"
        )

        print(
            "Real Estate Valuation Engine "
            "routing is working correctly."
        )

        print(
            "Strict input validation is "
            "compatible with inference."
        )

        print(
            "Target/leakage fields are "
            "excluded before prediction."
        )

    else:

        print()
        print(
            "⚠️ Some valuation engine tests failed."
        )

        raise SystemExit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

