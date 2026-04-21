# tabs/historical.py
"""
tabs/historical.py
Historical tab: period selector and load curves.
"""
import streamlit as st
import pandas as pd
import datetime

from utils.data_loader import load_featured_range
from utils.charts import load_curve


def render() -> None:
    """Render the historical page."""
    st.header("Historical data")

    # --- Period selector ---
    current_year = datetime.date.today().year
    year_options = list(range(2015, current_year + 1))

    col1, col2 = st.columns(2)
    with col1:
        year_start = st.selectbox("From", options=year_options, index=0)
    with col2:
        year_end = st.selectbox("To", options=year_options, index=len(year_options) - 1)

    if year_start > year_end:
        st.error("Start year must be before end year.")
        return

    # --- Load data ---
    try:
        df = load_featured_range(year_start, year_end)
    except FileNotFoundError as e:
        st.error(f"Historical data unavailable: {e}")
        return

    # --- Resample if range exceeds 3 months to keep rendering fast ---
    n_days = (year_end - year_start) * 365 + 365
    if n_days > 365:
        df_plot = (
            df.set_index("datetime")["load_t"]
            .resample("D")
            .mean()
            .reset_index()
            .rename(columns={"load_t": "load_MW"})
        )
        resolution_note = "Daily average"
    else:
        df_plot = df.rename(columns={"load_t": "load_MW"})
        resolution_note = "Hourly"

    # --- Metrics row ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Data points",
            value=f"{len(df):,}",
            help="Number of hourly observations loaded.",
        )
    with col2:
        mean_load = df["load_t"].mean() / 1000
        st.metric(
            label="Mean load",
            value=f"{mean_load:.2f} GW",
            help=f"Average over {year_start}–{year_end}.",
        )
    with col3:
        st.metric(
            label="Resolution",
            value=resolution_note,
            help="Hourly for periods under 3 months, daily average otherwise.",
        )

    st.divider()

    # --- Load curve ---
    fig = load_curve(
        df_plot,
        actual_col="load_MW",
        title=f"Electricity load {year_start}–{year_end} (GW)",
    )
    st.plotly_chart(fig, width='stretch')