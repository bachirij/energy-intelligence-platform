"""
dashboard.py - Entry point for the Streamlit dashboard.

Configures the page, renders the sidebar navigation, and routes
to the active page. Does not load data or compute anything directly.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st

# Configure the default settings of the page, must be Streamlit's first call
st.set_page_config(
    page_title="Energy Intelligence Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize state navigation
if "active_page" not in st.session_state:
    st.session_state.active_page = "realtime"

# Sidebar parameters
with st.sidebar:
    st.title("Energy Intelligence Platform")
    st.caption("France - h+1 forecast")
    st.divider()

    pages = {
        "realtime":   "Real-time",
        "historical": "Historical",
        "model_perf": "Model performance",
    }

    for key, label in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.active_page = key

    st.divider()
    st.caption("Data: ENTSO-E & Open-Meteo")
    st.caption("Model: XGBoost")

# Routing (import only the active page)
page = st.session_state.active_page

if page == "realtime":
    from tabs.realtime import render
    render()
elif page == "historical":
    from tabs.historical import render
    render()
elif page == "model_perf":
    from tabs.model_perf import render
    render()