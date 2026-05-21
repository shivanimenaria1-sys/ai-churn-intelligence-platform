"""
Dash layout components — sidebar, KPI cards, and page sections.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from dashboard.config import RISK_COLORS, THEME


def kpi_card(card_id: str, title: str, value_id: str, icon: str = "") -> dbc.Col:
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Span(icon, className="kpi-icon"),
                            html.P(title, className="kpi-title"),
                        ],
                        className="d-flex align-items-center gap-2",
                    ),
                    html.H3("—", id=value_id, className="kpi-value"),
                ]
            ),
            className="kpi-card shadow-sm",
            id=card_id,
        ),
        lg=3,
        md=6,
        sm=12,
        className="mb-3",
    )


def build_sidebar() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.H4("ChurnGuard", className="sidebar-brand"),
                    html.P("Executive Analytics", className="sidebar-sub"),
                ],
                className="sidebar-header",
            ),
            html.Hr(className="sidebar-divider"),
            dbc.Nav(
                [
                    dbc.NavLink("Overview", href="#", id="nav-overview", active=True),
                    dbc.NavLink("Risk Table", href="#", id="nav-table"),
                    dbc.NavLink("Trends", href="#", id="nav-trends"),
                    dbc.NavLink("Feature Importance", href="#", id="nav-importance"),
                    dbc.NavLink("Customer Deep Dive", href="#", id="nav-deepdive"),
                ],
                vertical=True,
                pills=True,
                className="sidebar-nav",
            ),
            html.Hr(className="sidebar-divider"),
            html.Div(id="api-status-badge", className="api-badge mt-3"),
        ],
        className="sidebar",
    )


def build_overview_section() -> html.Div:
    return html.Div(
        id="section-overview",
        children=[
            html.H2("Executive Overview", className="page-title"),
            html.P(
                "Real-time churn intelligence powered by XGBoost and FastAPI.",
                className="page-subtitle",
            ),
            dbc.Row(
                [
                    kpi_card("kpi-total", "Total Customers", "kpi-total-value", "👥"),
                    kpi_card("kpi-churn", "Avg Churn Risk", "kpi-churn-value", "📉"),
                    kpi_card("kpi-high", "High-Risk Customers", "kpi-high-value", "⚠️"),
                    kpi_card("kpi-revenue", "Revenue At Risk", "kpi-revenue-value", "💰"),
                ],
                className="mb-4",
            ),
            html.H4("Churn Risk Distribution", className="section-heading"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="graph-histogram"), lg=4, className="mb-3"),
                    dbc.Col(dcc.Graph(id="graph-pie"), lg=4, className="mb-3"),
                    dbc.Col(dcc.Graph(id="graph-risk-bar"), lg=4, className="mb-3"),
                ]
            ),
        ],
        className="content-section",
    )


def build_table_section() -> html.Div:
    return html.Div(
        id="section-table",
        style={"display": "none"},
        children=[
            html.H2("Customer Risk Register", className="page-title"),
            html.P("Sortable, filterable portfolio view for Customer Success prioritization.", className="page-subtitle"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id="table-search",
                            placeholder="Search customer ID, segment, contract…",
                            type="text",
                            className="mb-3",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="table-risk-filter",
                            options=[
                                {"label": "All Risk Tiers", "value": "all"},
                                {"label": "Low Risk", "value": "Low Risk"},
                                {"label": "Medium Risk", "value": "Medium Risk"},
                                {"label": "High Risk", "value": "High Risk"},
                            ],
                            value="all",
                            clearable=False,
                            className="mb-3",
                        ),
                        md=3,
                    ),
                ]
            ),
            dash_table.DataTable(
                id="customer-risk-table",
                page_size=15,
                sort_action="native",
                filter_action="native",
                page_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "textAlign": "left",
                    "padding": "10px",
                    "fontFamily": "Segoe UI, Roboto, sans-serif",
                    "fontSize": "13px",
                },
                style_header={
                    "backgroundColor": THEME["primary"],
                    "color": "white",
                    "fontWeight": "600",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{risk_category} = 'Low Risk'"},
                        "backgroundColor": "#d4edda",
                    },
                    {
                        "if": {"filter_query": "{risk_category} = 'Medium Risk'"},
                        "backgroundColor": "#fff3cd",
                    },
                    {
                        "if": {"filter_query": "{risk_category} = 'High Risk'"},
                        "backgroundColor": "#f8d7da",
                    },
                ],
            ),
        ],
        className="content-section",
    )


def build_trends_section() -> html.Div:
    return html.Div(
        id="section-trends",
        style={"display": "none"},
        children=[
            html.H2("Churn Trends & Segmentation", className="page-title"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="graph-segment"), lg=6, className="mb-3"),
                    dbc.Col(dcc.Graph(id="graph-contract"), lg=6, className="mb-3"),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="graph-tenure"), lg=6, className="mb-3"),
                    dbc.Col(dcc.Graph(id="graph-engagement"), lg=6, className="mb-3"),
                ]
            ),
        ],
        className="content-section",
    )


def build_importance_section() -> html.Div:
    return html.Div(
        id="section-importance",
        style={"display": "none"},
        children=[
            html.H2("Model Explainability", className="page-title"),
            html.P("XGBoost gain importance and SHAP global attribution.", className="page-subtitle"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="graph-xgb-importance"), lg=6, className="mb-3"),
                    dbc.Col(dcc.Graph(id="graph-shap-importance"), lg=6, className="mb-3"),
                ]
            ),
        ],
        className="content-section",
    )


def build_deepdive_section() -> html.Div:
    return html.Div(
        id="section-deepdive",
        style={"display": "none"},
        children=[
            html.H2("Customer Deep Dive", className="page-title"),
            html.P(
                "Select an account for live FastAPI scoring and risk factor breakdown.",
                className="page-subtitle",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Dropdown(id="customer-dropdown", placeholder="Select customer ID…"),
                            html.Div(id="live-api-result", className="mt-2"),
                        ],
                        lg=4,
                    ),
                    dbc.Col(dcc.Graph(id="graph-customer-gauge"), lg=4),
                    dbc.Col(html.Div(id="deepdive-risk-factors"), lg=4),
                ],
                className="mb-4",
            ),
            html.H5("Engagement Metrics"),
            html.Div(id="deepdive-engagement", className="metric-grid mb-3"),
            html.H5("Satisfaction Metrics"),
            html.Div(id="deepdive-satisfaction", className="metric-grid"),
        ],
        className="content-section",
    )


def build_layout() -> dbc.Container:
    """Full application layout."""
    return dbc.Container(
        [
            dcc.Store(id="active-page", data="overview"),
            dcc.Store(id="df-store"),
            dcc.Store(id="xgb-importance-store"),
            dcc.Store(id="shap-importance-store"),
            dcc.Interval(id="init-interval", interval=500, max_intervals=1),
            html.Div(
                [
                    build_sidebar(),
                    html.Div(
                        [
                            build_overview_section(),
                            build_table_section(),
                            build_trends_section(),
                            build_importance_section(),
                            build_deepdive_section(),
                        ],
                        className="main-content",
                    ),
                ],
                className="app-shell",
            ),
        ],
        fluid=True,
        className="dashboard-container p-0",
    )


# Risk legend for reference in CSS
RISK_LEGEND = html.Div(
    [
        html.Span("● Low", style={"color": RISK_COLORS["Low Risk"]}),
        html.Span("● Medium", style={"color": RISK_COLORS["Medium Risk"], "marginLeft": "12px"}),
        html.Span("● High", style={"color": RISK_COLORS["High Risk"], "marginLeft": "12px"}),
    ],
    className="risk-legend",
)
