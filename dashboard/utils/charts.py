"""
utils/charts.py — Reusable Plotly figures for the dashboard.

Each function takes data as input and returns a plotly.graph_objects.Figure.
No Streamlit calls here — rendering is handled by the pages via st.plotly_chart.
"""
import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------------------------
# Shared layout defaults (applied to every figure)
# ---------------------------------------------------------------------------

_LAYOUT = dict(
    font_family="sans-serif",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        zeroline=False,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
    ),
)

# Primary color used across all figures
_COLOR_ACTUAL    = "#4A90D9"
_COLOR_PREDICTED = "#E8834A"
_COLOR_DRIFT     = "#E24B4A"
_COLOR_OK        = "#639922"


# ---------------------------------------------------------------------------
# Load curve
# ---------------------------------------------------------------------------

def load_curve(
    df: pd.DataFrame,
    actual_col: str = "load_MW",
    predicted_col: str | None = None,
    title: str = "Electricity load (MW)",
) -> go.Figure:
    """Line chart of actual load, with an optional predicted series overlay.

    Args:
        df:            DataFrame with a 'datetime' column plus actual/predicted cols.
        actual_col:    Column name for actual load values.
        predicted_col: Optional column name for predicted load values.
        title:         Figure title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df[actual_col] / 1000,       # MW → GW
        name="Actual",
        line=dict(color=_COLOR_ACTUAL, width=1.5),
        mode="lines",
    ))

    if predicted_col and predicted_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=df[predicted_col] / 1000,  # MW → GW
            name="Predicted h+1",
            line=dict(color=_COLOR_PREDICTED, width=1.5, dash="dot"),
            mode="lines",
        ))

    fig.update_layout(
        **_LAYOUT,
        title=title,
        xaxis_title="Datetime (UTC)",
        yaxis_title="Load (GW)",        # MW → GW
    )
    return fig


def prediction_marker(
    fig: go.Figure,
    target_datetime: pd.Timestamp,
    predicted_load: float,
    last_datetime: pd.Timestamp,
    last_load: float,
) -> go.Figure:
    """Add a prediction point and a dashed connector to an existing load curve."""
    # Dashed connector between last actual and prediction
    fig.add_trace(go.Scatter(
        x=[last_datetime, target_datetime],
        y=[last_load / 1000, predicted_load / 1000],  # MW → GW
        mode="lines",
        line=dict(color=_COLOR_PREDICTED, width=1.5, dash="dot"),
        showlegend=False,
    ))
    # Prediction marker
    fig.add_trace(go.Scatter(
        x=[target_datetime],
        y=[predicted_load / 1000],                    # MW → GW
        name=f"Prediction: {predicted_load / 1000:.2f} GW",
        mode="markers",
        marker=dict(color=_COLOR_PREDICTED, size=10, symbol="circle"),
    ))
    return fig


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def feature_importance_chart(
    feature_cols: list[str],
    importances: list[float],
    title: str = "Feature importance",
) -> go.Figure:
    """Horizontal bar chart of XGBoost feature importances.

    Args:
        feature_cols:  List of feature names.
        importances:   Corresponding importance scores (same order).
        title:         Figure title.

    Returns:
        Plotly Figure.
    """
    pairs  = sorted(zip(feature_cols, importances), key=lambda x: x[1])
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=_COLOR_ACTUAL,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(**_LAYOUT, title=title, yaxis_title=None, height=320)
    fig.update_xaxes(tickformat=".0%", title_text="Importance score")
    return fig


# ---------------------------------------------------------------------------
# Drift report
# ---------------------------------------------------------------------------

def drift_bar_chart(
    drift_summary: dict,
    title: str = "Drift detection — latest report",
) -> go.Figure:
    """Bar chart of p-values per feature, with a threshold line at p=0.05.

    Args:
        drift_summary: dict from monitoring JSON, keyed by feature name.
                       Each value has 'p_value' and 'drift_detected'.
        title:         Figure title.

    Returns:
        Plotly Figure.
    """
    features = list(drift_summary.keys())
    p_values = [drift_summary[f]["p_value"] for f in features]
    colors   = [
        _COLOR_DRIFT if drift_summary[f]["drift_detected"] else _COLOR_OK
        for f in features
    ]

    fig = go.Figure(go.Bar(
        x=features,
        y=p_values,
        marker_color=colors,
        text=[f"{p:.3f}" for p in p_values],
        textposition="outside",
    ))

    fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="rgba(128,128,128,0.6)",
        annotation_text="threshold p=0.05",
        annotation_position="top right",
    )

    fig.update_layout(**_LAYOUT, title=title, height=340)
    fig.update_xaxes(title_text="Feature")
    fig.update_yaxes(
        title_text="p-value (K-S test)",
        range=[0, max(max(p_values) * 1.2, 0.12)],
    )
    return fig