"""
Dash callbacks — navigation, charts, table filters, and live API deep dive.
"""

from __future__ import annotations

import logging

import pandas as pd
from dash import Input, Output, State, html, no_update

from dashboard import figures
from dashboard.config import RISK_COLORS
from dashboard.data_service import (
    check_api_health,
    compute_kpis,
    get_cached_dashboard_data,
    get_customer_risk_factors,
    predict_via_api,
    row_to_payload,
)

logger = logging.getLogger(__name__)

TABLE_COLUMNS = [
    {"name": "Customer ID", "id": "customer_id"},
    {"name": "Churn Probability", "id": "churn_probability"},
    {"name": "Risk Category", "id": "risk_category"},
    {"name": "Revenue", "id": "total_revenue", "type": "numeric", "format": {"specifier": "$,.0f"}},
    {"name": "Support Tickets", "id": "support_tickets", "type": "numeric"},
    {"name": "CSAT", "id": "csat_score", "type": "numeric", "format": {"specifier": ".1f"}},
    {"name": "Segment", "id": "customer_segment"},
    {"name": "Contract", "id": "contract_type"},
]


def register_callbacks(app):
    """Attach all dashboard callbacks to the Dash app."""

    @app.callback(
        Output("df-store", "data"),
        Output("xgb-importance-store", "data"),
        Output("shap-importance-store", "data"),
        Output("api-status-badge", "children"),
        Output("api-status-badge", "className"),
        Input("init-interval", "n_intervals"),
        prevent_initial_call=False,
    )
    def load_initial_data(_):
        api_up = check_api_health()
        badge_class = "api-badge online" if api_up else "api-badge offline"
        badge_text = "● FastAPI Live" if api_up else "● API Offline (local scoring)"

        try:
            df, xgb_imp, shap_imp = get_cached_dashboard_data()
            logger.info("Dashboard data ready: %d customers", len(df))
            return (
                df.to_dict("records"),
                xgb_imp.to_dict("records"),
                shap_imp.to_dict("records"),
                badge_text,
                badge_class,
            )
        except Exception as exc:
            logger.exception("Failed to load dashboard data")
            return [], [], [], f"● Load Error: {exc}", "api-badge offline"

    @app.callback(
        Output("section-overview", "style"),
        Output("section-table", "style"),
        Output("section-trends", "style"),
        Output("section-importance", "style"),
        Output("section-deepdive", "style"),
        Output("nav-overview", "active"),
        Output("nav-table", "active"),
        Output("nav-trends", "active"),
        Output("nav-importance", "active"),
        Output("nav-deepdive", "active"),
        Output("active-page", "data"),
        Input("nav-overview", "n_clicks"),
        Input("nav-table", "n_clicks"),
        Input("nav-trends", "n_clicks"),
        Input("nav-importance", "n_clicks"),
        Input("nav-deepdive", "n_clicks"),
        State("active-page", "data"),
        prevent_initial_call=True,
    )
    def navigate(ov, tb, tr, im, dd, current):
        ctx = __import__("dash").callback_context
        if not ctx.triggered:
            return (no_update,) * 10
        nav_id = ctx.triggered[0]["prop_id"].split(".")[0]
        page_map = {
            "nav-overview": "overview",
            "nav-table": "table",
            "nav-trends": "trends",
            "nav-importance": "importance",
            "nav-deepdive": "deepdive",
        }
        page = page_map.get(nav_id, current or "overview")
        show = {"display": "block"}
        hide = {"display": "none"}
        styles = {
            "overview": [show, hide, hide, hide, hide],
            "table": [hide, show, hide, hide, hide],
            "trends": [hide, hide, show, hide, hide],
            "importance": [hide, hide, hide, show, hide],
            "deepdive": [hide, hide, hide, hide, show],
        }
        s = styles[page]
        actives = [p == page for p in ["overview", "table", "trends", "importance", "deepdive"]]
        return (*s, *actives, page)

    @app.callback(
        Output("kpi-total-value", "children"),
        Output("kpi-churn-value", "children"),
        Output("kpi-high-value", "children"),
        Output("kpi-revenue-value", "children"),
        Output("graph-histogram", "figure"),
        Output("graph-pie", "figure"),
        Output("graph-risk-bar", "figure"),
        Input("df-store", "data"),
    )
    def update_overview(data):
        if not data:
            loading = figures.empty_figure("Churn Probability Distribution", "Loading customer data…")
            return "—", "—", "—", "—", loading, loading, loading
        df = pd.DataFrame(data)
        kpis = compute_kpis(df)
        return (
            f"{kpis['total_customers']:,}",
            f"{kpis['churn_rate']:.1%}",
            f"{kpis['high_risk_customers']:,}",
            f"${kpis['revenue_at_risk']:,.0f}",
            figures.churn_probability_histogram(df),
            figures.risk_pie_chart(df),
            figures.risk_bar_chart(df),
        )

    @app.callback(
        Output("customer-risk-table", "data"),
        Output("customer-risk-table", "columns"),
        Input("df-store", "data"),
        Input("table-search", "value"),
        Input("table-risk-filter", "value"),
    )
    def update_table(data, search, risk_filter):
        if not data:
            return [], TABLE_COLUMNS
        df = pd.DataFrame(data)
        if risk_filter and risk_filter != "all":
            df = df[df["risk_category"] == risk_filter]
        if search:
            mask = df.astype(str).apply(
                lambda row: row.str.contains(search, case=False, na=False).any(),
                axis=1,
            )
            df = df[mask]
        display = df[
            [
                "customer_id",
                "churn_probability",
                "risk_category",
                "total_revenue",
                "support_tickets",
                "csat_score",
                "customer_segment",
                "contract_type",
            ]
        ].copy()
        display["churn_probability"] = (display["churn_probability"] * 100).round(2).astype(str) + "%"
        return display.to_dict("records"), TABLE_COLUMNS

    @app.callback(
        Output("graph-segment", "figure"),
        Output("graph-contract", "figure"),
        Output("graph-tenure", "figure"),
        Output("graph-engagement", "figure"),
        Input("df-store", "data"),
    )
    def update_trends(data):
        if not data:
            loading = figures.empty_figure("Chart", "Loading trend data…")
            return loading, loading, loading, loading
        df = pd.DataFrame(data)
        return (
            figures.churn_by_segment(df),
            figures.churn_by_contract(df),
            figures.churn_vs_tenure(df),
            figures.churn_vs_engagement(df),
        )

    @app.callback(
        Output("graph-xgb-importance", "figure"),
        Output("graph-shap-importance", "figure"),
        Input("xgb-importance-store", "data"),
        Input("shap-importance-store", "data"),
    )
    def update_importance(xgb_data, shap_data):
        if not xgb_data:
            loading = figures.empty_figure("Feature Importance", "Loading model explainability…")
            return loading, loading
        return (
            figures.xgboost_importance_chart(pd.DataFrame(xgb_data)),
            figures.shap_importance_chart(pd.DataFrame(shap_data)),
        )

    @app.callback(
        Output("customer-dropdown", "options"),
        Output("customer-dropdown", "value"),
        Input("df-store", "data"),
    )
    def populate_customers(data):
        if not data:
            return [], None
        df = pd.DataFrame(data)
        high = df[df["risk_category"] == "High Risk"].head(1)
        default = high["customer_id"].iloc[0] if len(high) else df["customer_id"].iloc[0]
        options = [{"label": cid, "value": cid} for cid in df["customer_id"].head(500)]
        return options, default

    @app.callback(
        Output("graph-customer-gauge", "figure"),
        Output("deepdive-risk-factors", "children"),
        Output("deepdive-engagement", "children"),
        Output("deepdive-satisfaction", "children"),
        Output("live-api-result", "children"),
        Input("customer-dropdown", "value"),
        Input("df-store", "data"),
        Input("xgb-importance-store", "data"),
    )
    def update_deepdive(customer_id, data, xgb_data):
        if not customer_id or not data:
            return figures.customer_gauge(0), "", "", "", ""

        df = pd.DataFrame(data)
        row = df[df["customer_id"] == customer_id].iloc[0]

        # Live FastAPI prediction
        api_result = predict_via_api(row_to_payload(row))
        if api_result:
            proba = api_result["churn_probability"]
            risk = api_result["risk_category"]
            source = "FastAPI /predict (live)"
        else:
            proba = row["churn_probability"]
            risk = row["risk_category"]
            source = "Cached/local score"

        importance = pd.DataFrame(xgb_data) if xgb_data else pd.DataFrame()
        factors = get_customer_risk_factors(row, importance)
        factor_ui = [
            html.Div(f"{f['metric']}: {f['value']}", className="risk-factor-item")
            for f in factors
        ]

        engagement = html.Div(
            [
                _metric_tile("Monthly Logins", row["monthly_logins"]),
                _metric_tile("Weekly Active Days", row["weekly_active_days"]),
                _metric_tile("Avg Session (min)", f"{row['avg_session_time']:.1f}"),
                _metric_tile("Features Used", row["features_used"]),
                _metric_tile("Usage Growth", f"{row['usage_growth_rate']:.2%}"),
            ],
            className="metric-grid",
        )

        satisfaction = html.Div(
            [
                _metric_tile("CSAT Score", f"{row['csat_score']:.1f}"),
                _metric_tile("NPS Score", f"{row['nps_score']:.1f}"),
                _metric_tile("Support Tickets", row["support_tickets"]),
                _metric_tile("Payment Failures", row["payment_failures"]),
            ],
            className="metric-grid",
        )

        live_banner = html.Div(
            [html.Strong(f"Risk: {risk}"), f" — P(churn)={proba:.1%} | Source: {source}"],
            className="live-api-banner",
            style={"borderColor": RISK_COLORS.get(risk, "#ccc")},
        )

        return (
            figures.customer_gauge(proba),
            factor_ui,
            engagement,
            satisfaction,
            live_banner,
        )


def _metric_tile(label: str, value) -> html.Div:
    return html.Div(
        [html.Div(label, className="label"), html.Div(str(value), className="value")],
        className="metric-tile",
    )
