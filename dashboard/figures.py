"""
Plotly figure builders for the churn executive dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config import RISK_COLORS, THEME


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        font=dict(family="Segoe UI, Roboto, Helvetica, Arial", color=THEME["text"], size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=24, t=48, b=40),
        hovermode="closest",
    )
    return fig


def empty_figure(title: str, message: str = "Loading data…") -> go.Figure:
    """Placeholder chart while data is loading or unavailable."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color=THEME["muted"]),
    )
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=320,
    )
    return _apply_theme(fig)


def churn_probability_histogram(df: pd.DataFrame) -> go.Figure:
    if df.empty or "churn_probability" not in df.columns or len(df) < 2:
        return empty_figure("Churn Probability Distribution")
    fig = px.histogram(
        df,
        x="churn_probability",
        nbins=40,
        color_discrete_sequence=[THEME["secondary"]],
        labels={"churn_probability": "Churn Probability"},
        title="Churn Probability Distribution",
    )
    return _apply_theme(fig)


def risk_pie_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or "risk_category" not in df.columns:
        return empty_figure("Customers by Risk Tier")
    counts = df["risk_category"].value_counts().reset_index()
    counts.columns = ["risk_category", "count"]
    fig = px.pie(
        counts,
        names="risk_category",
        values="count",
        color="risk_category",
        color_discrete_map=RISK_COLORS,
        title="Customers by Risk Tier",
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_theme(fig)


def risk_bar_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or "risk_category" not in df.columns:
        return empty_figure("Risk Tier Counts")
    order = ["Low Risk", "Medium Risk", "High Risk"]
    counts = df["risk_category"].value_counts().reindex(order).fillna(0).reset_index()
    counts.columns = ["risk_category", "count"]
    fig = px.bar(
        counts,
        x="risk_category",
        y="count",
        color="risk_category",
        color_discrete_map=RISK_COLORS,
        title="Risk Tier Counts",
        labels={"count": "Customers", "risk_category": "Risk Tier"},
    )
    return _apply_theme(fig)


def churn_by_segment(df: pd.DataFrame) -> go.Figure:
    if df.empty or "customer_segment" not in df.columns:
        return empty_figure("Avg Churn Probability by Segment")
    agg = df.groupby("customer_segment").agg(
        churn_rate=("churn_probability", "mean"),
        customers=("customer_id", "count"),
    ).reset_index()
    fig = px.bar(
        agg,
        x="customer_segment",
        y="churn_rate",
        color="customer_segment",
        text=agg["churn_rate"].map(lambda x: f"{x:.1%}"),
        title="Avg Churn Probability by Segment",
        labels={"churn_rate": "Avg Churn Probability"},
    )
    fig.update_traces(textposition="outside")
    return _apply_theme(fig)


def churn_by_contract(df: pd.DataFrame) -> go.Figure:
    if df.empty or "contract_type" not in df.columns:
        return empty_figure("Avg Churn Probability by Contract Type")
    agg = df.groupby("contract_type").agg(
        churn_rate=("churn_probability", "mean"),
    ).reset_index()
    fig = px.bar(
        agg,
        x="contract_type",
        y="churn_rate",
        color="contract_type",
        color_discrete_sequence=px.colors.sequential.Blues_r,
        title="Avg Churn Probability by Contract Type",
    )
    return _apply_theme(fig)


def churn_vs_tenure(df: pd.DataFrame) -> go.Figure:
    if df.empty or "tenure_months" not in df.columns:
        return empty_figure("Churn Probability vs Tenure")
    fig = px.scatter(
        df.sample(min(1500, len(df)), random_state=42),
        x="tenure_months",
        y="churn_probability",
        color="risk_category",
        color_discrete_map=RISK_COLORS,
        opacity=0.55,
        title="Churn Probability vs Tenure",
        labels={"tenure_months": "Tenure (months)", "churn_probability": "Churn Probability"},
        hover_data=["customer_id", "total_revenue"],
    )
    return _apply_theme(fig)


def churn_vs_engagement(df: pd.DataFrame) -> go.Figure:
    if df.empty or "monthly_logins" not in df.columns:
        return empty_figure("Churn Probability vs Engagement Index")
    df = df.copy()
    df["engagement_index"] = (
        df["monthly_logins"] / (df["monthly_logins"].max() + 1)
        + df["weekly_active_days"] / 7
        + df["features_used"] / (df["features_used"].max() + 1)
    ) / 3
    fig = px.scatter(
        df.sample(min(1500, len(df)), random_state=42),
        x="engagement_index",
        y="churn_probability",
        color="risk_category",
        color_discrete_map=RISK_COLORS,
        opacity=0.55,
        title="Churn Probability vs Engagement Index",
        labels={"engagement_index": "Engagement Index (0–1)"},
    )
    return _apply_theme(fig)


def xgboost_importance_chart(importance: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = importance.head(top_n).sort_values("importance")
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color_discrete_sequence=[THEME["primary"]],
        title="XGBoost Feature Importance (Gain)",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    return _apply_theme(fig)


def shap_importance_chart(shap_imp: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = shap_imp.head(top_n).sort_values("mean_abs_shap")
    fig = px.bar(
        top,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color_discrete_sequence=[THEME["secondary"]],
        title="SHAP Global Importance (Mean |SHAP|)",
        labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
    )
    return _apply_theme(fig)


def customer_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": "Churn Risk Score", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": THEME["primary"]},
                "steps": [
                    {"range": [0, 33], "color": "#d4edda"},
                    {"range": [33, 66], "color": "#fff3cd"},
                    {"range": [66, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": RISK_COLORS["High Risk"], "width": 4},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return _apply_theme(fig)
