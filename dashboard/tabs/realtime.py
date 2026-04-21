# pages/realtime.py
"""
pages/realtime.py 

Real-time tab: current day load curve and h+1 prediction.
"""
import streamlit as st
import pandas as pd

from utils.data_loader import load_realtime
from utils.prediction import predict_next_hour
from utils.charts import load_curve, prediction_marker


def render() -> None:
    """Render the real-time page."""
    st.header("Real-time forecast")

    # --- Load data ---
    try:
        df = load_realtime()
    except FileNotFoundError as e:
        st.error(f"Realtime data unavailable: {e}")
        return

    try:
        prediction = predict_next_hour()
    except (FileNotFoundError, ValueError) as e:
        st.warning(f"Prediction unavailable: {e}")
        prediction = None

    # --- Metrics row ---
    last_actual = df["load_MW"].iloc[-1]
    last_dt     = df["datetime"].iloc[-1]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Last actual load",
            value=f"{last_actual:,.0f} MW",
            help=f"Observed at {last_dt.strftime('%H:%M UTC')}",
        )
    with col2:
        if prediction:
            st.metric(
                label="Predicted load h+1",
                value=f"{prediction['predicted_load_MW']:,.0f} MW",
                delta=f"{prediction['predicted_load_MW'] - last_actual:+,.0f} MW",
                help=f"Target: {prediction['target_datetime'].strftime('%H:%M UTC')}",
            )
        else:
            st.metric(label="Predicted load h+1", value="N/A")
    with col3:
        if prediction:
            st.metric(
                label="Forecast horizon",
                value=prediction["target_datetime"].strftime("%H:%M UTC"),
                help="Target datetime of the prediction (t+1)",
            )
        else:
            st.metric(label="Forecast horizon", value="N/A")

    st.divider()

    # --- Load curve: last 24 hours only ---
    now_utc  = pd.Timestamp.now(tz="UTC")
    cutoff = last_dt - pd.Timedelta(hours=24)
    df_today = df[df["datetime"] >= cutoff].copy()

    if df_today.empty:
        st.warning("Not enough realtime data to display the current day curve.")
        return

    fig = load_curve(
        df_today,
        actual_col="load_MW",
        title="Electricity load - last 24 hours (MW)",
    )

    if prediction:
        fig = prediction_marker(
            fig,
            target_datetime=prediction["target_datetime"],
            predicted_load=prediction["predicted_load_MW"],
            last_datetime=last_dt,
            last_load=last_actual,
        )

    st.plotly_chart(fig, width='stretch')

    # --- MLOps expander: feature values used for prediction ---
    if prediction:
        with st.expander("Feature values used for this prediction"):
            feat = prediction["feature_values"]
            c1, c2 = st.columns(2)
            with c1:
                st.metric("load_t",     f"{feat['load_t']:,.0f} MW")
                st.metric("load_t-1",   f"{feat['load_t-1']:,.0f} MW")
                st.metric("load_t-24",  f"{feat['load_t-24']:,.0f} MW")
                st.metric("load_t-168", f"{feat['load_t-168']:,.0f} MW")
                st.metric("temperature_t", f"{feat['temperature_t']:.1f} °C")
            with c2:
                st.metric("temperature_t-24", f"{feat['temperature_t-24']:.1f} °C")
                st.metric("hour",         str(feat["hour"]))
                st.metric("is_weekday",   str(feat["is_weekday"]))
                st.metric("day_of_week",  str(feat["day_of_week"]))
                st.metric("week_of_year", str(feat["week_of_year"]))