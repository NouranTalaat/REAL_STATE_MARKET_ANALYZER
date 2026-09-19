from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Egypt Real Estate Market Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .dashboard-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=300)
def load_market_overview() -> pd.DataFrame:
    from src.sql_analytics import load_market_overview

    return load_market_overview()


@st.cache_data(ttl=300)
def load_market_summary() -> pd.DataFrame:
    from src.sql_analytics import load_market_summary

    return load_market_summary()


@st.cache_data(ttl=300)
def load_robust_metrics() -> pd.DataFrame:
    from src.sql_analytics import load_robust_metrics

    return load_robust_metrics()


@st.cache_data(ttl=300)
def load_location_intelligence() -> pd.DataFrame:
    from src.sql_analytics import load_location_intelligence

    return load_location_intelligence()


@st.cache_data(ttl=300)
def load_city_market_profile() -> pd.DataFrame:
    from src.sql_analytics import load_city_market_profile

    return load_city_market_profile()


@st.cache_data(ttl=300)
def load_opportunity_signals() -> pd.DataFrame:
    from src.sql_analytics import load_opportunity_signals

    return load_opportunity_signals()


@st.cache_data(ttl=300)
def load_data_quality() -> pd.DataFrame:
    from src.sql_analytics import load_data_quality_monitor

    return load_data_quality_monitor()


# ============================================================
# SAFE COLUMN HELPERS
# ============================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    normalized = {
        str(column).lower().strip(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


def numeric_value(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float | None:

    column = find_column(dataframe, candidates)

    if column is None:
        return None

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.iloc[0])


def format_number(value: float | None) -> str:

    if value is None:
        return "N/A"

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def format_currency(value: float | None) -> str:

    if value is None:
        return "N/A"

    if abs(value) >= 1_000_000:
        return f"EGP {value / 1_000_000:.2f}M"

    if abs(value) >= 1_000:
        return f"EGP {value / 1_000:.1f}K"

    return f"EGP {value:,.0f}"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏠 Market Intelligence")

st.sidebar.markdown(
    """
    **Egyptian Real Estate Market**

    Interactive analytics dashboard powered by
    the project's SQL analytics layer.
    """
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Market Overview",
        "Location Intelligence",
        "Opportunity Signals",
        "Data Quality",
    ],
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    "Egypt Real Estate Market Intelligence"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    "Market analytics, pricing intelligence, location insights, "
    "and opportunity signals."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# MARKET OVERVIEW DATA
# ============================================================

try:

    market_overview = load_market_overview()
    market = load_market_summary()

except Exception as error:

    st.error(
        "Unable to connect to the analytics database."
    )

    st.code(str(error))

    st.info(
        "Make sure SQL Server is running and the configured "
        "database is available."
    )

    st.stop()


# ============================================================
# MARKET OVERVIEW
# ============================================================

if page == "Market Overview":

    st.header("📊 Market Overview")

    # --------------------------------------------------------
    # OVERALL MARKET KPI VALUES
    # --------------------------------------------------------

    if market_overview.empty:

        st.warning(
            "No overall market overview data is available."
        )

    else:

        overview = market_overview.iloc[0]

        total_listings = pd.to_numeric(
            overview.get("total_listings"),
            errors="coerce",
        )

        average_price = pd.to_numeric(
            overview.get("avg_price_egp"),
            errors="coerce",
        )

        median_price = pd.to_numeric(
            overview.get("median_price_egp"),
            errors="coerce",
        )

        avg_price_per_sqm = pd.to_numeric(
            overview.get("avg_price_per_sqm"),
            errors="coerce",
        )

        sale_listings = pd.to_numeric(
            overview.get("sale_listings"),
            errors="coerce",
        )

        rent_listings = pd.to_numeric(
            overview.get("rent_listings"),
            errors="coerce",
        )

        # ----------------------------------------------------
        # KPI CARDS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Listings",
                format_number(
                    float(total_listings)
                    if pd.notna(total_listings)
                    else None
                ),
            )

        with col2:

            st.metric(
                "Median Price",
                format_currency(
                    float(median_price)
                    if pd.notna(median_price)
                    else None
                ),
            )

        with col3:

            st.metric(
                "Average Price / sqm",
                format_currency(
                    float(avg_price_per_sqm)
                    if pd.notna(avg_price_per_sqm)
                    else None
                ),
            )

        with col4:

            st.metric(
                "Average Price",
                format_currency(
                    float(average_price)
                    if pd.notna(average_price)
                    else None
                ),
            )

        st.divider()

        # ----------------------------------------------------
        # MARKET MIX
        # ----------------------------------------------------

        st.subheader("Market Composition")

        mix_col1, mix_col2 = st.columns(2)

        with mix_col1:

            st.metric(
                "Sale Listings",
                format_number(
                    float(sale_listings)
                    if pd.notna(sale_listings)
                    else None
                ),
            )

        with mix_col2:

            st.metric(
                "Rent Listings",
                format_number(
                    float(rent_listings)
                    if pd.notna(rent_listings)
                    else None
                ),
            )

        # ----------------------------------------------------
        # MARKET DATA
        # ----------------------------------------------------

        if market.empty:

            st.warning(
                "No market summary data is available."
            )

        else:

            st.divider()

            st.subheader("Market Summary")

            st.dataframe(
                market,
                width="stretch",
                hide_index=True,
            )

            # ------------------------------------------------
            # NUMERIC VISUALIZATION
            # ------------------------------------------------

            numeric_columns = market.select_dtypes(
                include="number"
            ).columns.tolist()

            categorical_columns = market.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()

            if numeric_columns and categorical_columns:

                category_column = categorical_columns[0]

                selected_metric = st.selectbox(
                    "Select market metric",
                    numeric_columns,
                )

                chart_data = market[
                    [category_column, selected_metric]
                ].dropna()

                if not chart_data.empty:

                    chart = px.bar(
                        chart_data,
                        x=category_column,
                        y=selected_metric,
                        title=(
                            f"{selected_metric} "
                            f"by {category_column}"
                        ),
                    )

                    chart.update_layout(
                        xaxis_title=category_column,
                        yaxis_title=selected_metric,
                    )

                    st.plotly_chart(
                        chart,
                        width="stretch",
                    )


# ============================================================
# LOCATION INTELLIGENCE
# ============================================================

elif page == "Location Intelligence":

    st.header("📍 Location Intelligence")

    try:

        locations = load_location_intelligence()
        city_profiles = load_city_market_profile()

    except Exception as error:

        st.error(
            "Unable to load location analytics."
        )

        st.code(str(error))

        st.stop()

    # --------------------------------------------------------
    # LOCATION TABLE
    # --------------------------------------------------------

    if not locations.empty:

        st.subheader("Location Intelligence")

        st.dataframe(
            locations,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # DYNAMIC LOCATION CHART
        # ----------------------------------------------------

        categorical_columns = locations.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        numeric_columns = locations.select_dtypes(
            include="number"
        ).columns.tolist()

        if categorical_columns and numeric_columns:

            location_column = st.selectbox(
                "Location dimension",
                categorical_columns,
            )

            metric_column = st.selectbox(
                "Metric",
                numeric_columns,
            )

            chart_data = locations[
                [location_column, metric_column]
            ].dropna()

            chart_data = (
                chart_data
                .sort_values(
                    metric_column,
                    ascending=False,
                )
                .head(15)
            )

            chart = px.bar(
                chart_data,
                x=metric_column,
                y=location_column,
                orientation="h",
                title=f"Top Locations by {metric_column}",
            )

            chart.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                chart,
                width="stretch",
            )

    # --------------------------------------------------------
    # CITY MARKET PROFILE
    # --------------------------------------------------------

    if not city_profiles.empty:

        st.divider()

        st.subheader("🏙️ City Market Profiles")

        st.dataframe(
            city_profiles,
            width="stretch",
            hide_index=True,
        )

        city_columns = city_profiles.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        numeric_columns = city_profiles.select_dtypes(
            include="number"
        ).columns.tolist()

        if city_columns and numeric_columns:

            city_column = city_columns[0]

            metric_column = st.selectbox(
                "City comparison metric",
                numeric_columns,
                key="city_metric",
            )

            chart_data = city_profiles[
                [city_column, metric_column]
            ].dropna()

            chart_data = (
                chart_data
                .sort_values(
                    metric_column,
                    ascending=False,
                )
                .head(15)
            )

            chart = px.bar(
                chart_data,
                x=city_column,
                y=metric_column,
                title=f"City Comparison — {metric_column}",
            )

            chart.update_layout(
                xaxis_tickangle=-45,
            )

            st.plotly_chart(
                chart,
                width="stretch",
            )


# ============================================================
# OPPORTUNITY SIGNALS
# ============================================================

elif page == "Opportunity Signals":

    st.header("💡 Market Opportunity Signals")

    try:

        opportunities = load_opportunity_signals()

    except Exception as error:

        st.error(
            "Unable to load opportunity signals."
        )

        st.code(str(error))

        st.stop()

    if opportunities.empty:

        st.info(
            "No opportunity signals are currently available."
        )

    else:

        st.subheader(
            "Potential Market Opportunities"
        )

        st.dataframe(
            opportunities,
            width="stretch",
            hide_index=True,
        )

        categorical_columns = opportunities.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        numeric_columns = opportunities.select_dtypes(
            include="number"
        ).columns.tolist()

        if categorical_columns and numeric_columns:

            category_column = st.selectbox(
                "Opportunity dimension",
                categorical_columns,
            )

            metric_column = st.selectbox(
                "Opportunity metric",
                numeric_columns,
            )

            chart_data = opportunities[
                [category_column, metric_column]
            ].dropna()

            chart_data = (
                chart_data
                .sort_values(
                    metric_column,
                    ascending=False,
                )
                .head(15)
            )

            chart = px.bar(
                chart_data,
                x=category_column,
                y=metric_column,
                title=f"Opportunity Signals — {metric_column}",
            )

            chart.update_layout(
                xaxis_tickangle=-45,
            )

            st.plotly_chart(
                chart,
                width="stretch",
            )


# ============================================================
# DATA QUALITY
# ============================================================

elif page == "Data Quality":

    st.header("🛡️ Data Quality Monitor")

    try:

        quality = load_data_quality()

    except Exception as error:

        st.error(
            "Unable to load data quality monitoring."
        )

        st.code(str(error))

        st.stop()

    if quality.empty:

        st.info(
            "No data quality monitoring records are available."
        )

    else:

        st.dataframe(
            quality,
            width="stretch",
            hide_index=True,
        )

        numeric_columns = quality.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = quality.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        if numeric_columns:

            st.subheader("Quality Metrics")

            metric = st.selectbox(
                "Quality metric",
                numeric_columns,
                key="quality_metric",
            )

            values = pd.to_numeric(
                quality[metric],
                errors="coerce",
            ).dropna()

            if not values.empty:

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=list(range(len(values))),
                        y=values,
                        name=metric,
                    )
                )

                fig.update_layout(
                    title=f"Data Quality — {metric}",
                    xaxis_title="Record",
                    yaxis_title=metric,
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

        if categorical_columns:

            st.subheader("Quality Status")

            status_column = categorical_columns[0]

            status_counts = (
                quality[status_column]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            status_counts.columns = [
                status_column,
                "count",
            ]

            fig = px.pie(
                status_counts,
                names=status_column,
                values="count",
                title="Quality Status Distribution",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Real Estate Market Intelligence Platform | "
    "Python • SQL Server • Pandas • Plotly • Streamlit"
)