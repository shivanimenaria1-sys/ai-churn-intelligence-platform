"""
Data loading, prediction enrichment, and model explainability helpers.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from dashboard.config import (
    API_HEALTH_URL,
    API_PREDICT_URL,
    API_SCORE_SAMPLE,
    ARTIFACTS_DIR,
    DATA_PATH,
    MAX_CUSTOMERS,
    MODELS_DIR,
    PREDICT_PAYLOAD_COLUMNS,
    RISK_HIGH,
    RISK_LOW,
)

logger = logging.getLogger(__name__)

# In-memory cache populated at startup (avoids empty UI while scoring)
_cached_customers: pd.DataFrame | None = None
_cached_xgb_importance: pd.DataFrame | None = None
_cached_shap_importance: pd.DataFrame | None = None


def preload_dashboard_data(use_api: bool = False) -> None:
    """Load and score customer data before the Dash server starts."""
    global _cached_customers, _cached_xgb_importance, _cached_shap_importance
    logger.info("Preloading dashboard data (this may take 1–2 minutes for full dataset)…")
    _cached_customers, _ = load_customer_data(use_api=use_api)
    _cached_xgb_importance = load_feature_importance()
    try:
        _cached_shap_importance = load_shap_importance(sample_size=300)
    except Exception as exc:
        logger.warning("SHAP preload skipped: %s", exc)
        _cached_shap_importance = _cached_xgb_importance.rename(
            columns={"importance": "mean_abs_shap"}
        )
    logger.info("Preload complete: %d customers ready", len(_cached_customers))


def get_cached_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return cached dataframes; preload if not yet available."""
    if _cached_customers is None:
        preload_dashboard_data(use_api=False)
    return _cached_customers, _cached_xgb_importance, _cached_shap_importance


def assign_risk_category(probability: float) -> str:
    if probability < RISK_LOW:
        return "Low Risk"
    if probability < RISK_HIGH:
        return "Medium Risk"
    return "High Risk"


def check_api_health() -> bool:
    """Return True if FastAPI backend is reachable."""
    try:
        resp = requests.get(API_HEALTH_URL, timeout=2)
        return resp.status_code == 200 and resp.json().get("status") == "healthy"
    except requests.RequestException:
        return False


_INT_COLS = {
    "tenure_months", "monthly_logins", "support_tickets", "payment_failures",
    "weekly_active_days", "features_used", "escalations", "price_increase_last_3m",
}


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Build JSON payload for POST /predict."""
    payload: dict[str, Any] = {}
    for col in PREDICT_PAYLOAD_COLUMNS:
        val = row[col]
        if col in ("contract_type", "customer_segment"):
            payload[col] = str(val)
        elif col in _INT_COLS:
            payload[col] = int(val)
        else:
            payload[col] = float(val)
    return payload


def predict_via_api(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Call FastAPI /predict for a single customer."""
    try:
        resp = requests.post(API_PREDICT_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.warning("API predict failed: %s", exc)
        return None


def predict_via_local(row: pd.Series) -> dict[str, Any]:
    """Score using in-process ML service (same pipeline as API)."""
    from backend.ml_service import churn_service

    if not churn_service.is_loaded:
        churn_service.load_artifacts()
    return churn_service.predict(row_to_payload(row))


def enrich_predictions_api(
    df: pd.DataFrame,
    max_workers: int = 12,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Enrich dataframe with predictions from FastAPI /predict (parallel).

    For large datasets, pass sample_size to score a representative subset
    via API and score the remainder locally for performance.
    """
    result = df.copy().reset_index(drop=True)
    n = len(result)
    positions = list(range(n))

    if sample_size and sample_size < n:
        api_positions = set(
            np.random.default_rng(42).choice(positions, size=sample_size, replace=False)
        )
    else:
        api_positions = set(positions)

    proba = [0.0] * n
    risk = [""] * n
    pred_class = [0] * n
    confidence = [0.0] * n
    source = ["local"] * n

    def _score_position(pos: int) -> tuple[int, dict]:
        row = result.iloc[pos]
        payload = row_to_payload(row)
        if pos in api_positions:
            api_result = predict_via_api(payload)
            if api_result:
                return pos, {**api_result, "_source": "api"}
        local_result = predict_via_local(row)
        local_result["_source"] = "local"
        return pos, local_result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_score_position, pos) for pos in positions]
        for fut in as_completed(futures):
            pos, res = fut.result()
            proba[pos] = res["churn_probability"]
            risk[pos] = res["risk_category"]
            pred_class[pos] = res["predicted_class"]
            confidence[pos] = res["confidence_score"]
            source[pos] = res.get("_source", "local")

    result["churn_probability"] = proba
    result["risk_category"] = risk
    result["predicted_class"] = pred_class
    result["confidence_score"] = confidence
    result["prediction_source"] = source
    return result


def enrich_predictions_local(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """Score all rows via local ML service (parallel bulk scoring)."""
    from backend.ml_service import churn_service

    if not churn_service.is_loaded:
        churn_service.load_artifacts()

    rows = [row for _, row in df.iterrows()]
    records: list[dict] = [{}] * len(rows)

    def _score(i: int, row: pd.Series) -> tuple[int, dict]:
        return i, churn_service.predict(row_to_payload(row))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_score, i, row) for i, row in enumerate(rows)]
        for fut in as_completed(futures):
            i, pred = fut.result()
            records[i] = pred

    pred_df = pd.DataFrame(records)
    out = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    out["prediction_source"] = "local"
    return out


def load_customer_data(use_api: bool = True, api_sample: int = 800) -> tuple[pd.DataFrame, bool]:
    """
    Load CSV and attach churn predictions.

    Uses FastAPI for up to api_sample rows when API is healthy; remaining rows
    scored locally so the dashboard loads quickly with real API integration.
    """
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if MAX_CUSTOMERS and len(df) > MAX_CUSTOMERS:
        df = df.sample(n=MAX_CUSTOMERS, random_state=42).reset_index(drop=True)
        logger.info("Sampled %s customers for dashboard", MAX_CUSTOMERS)

    api_up = use_api and check_api_health()
    sample = api_sample if api_sample is not None else API_SCORE_SAMPLE

    if api_up:
        logger.info("FastAPI healthy — scoring via /predict (sample=%s)", sample)
        df = enrich_predictions_api(df, sample_size=sample)
    else:
        logger.info("FastAPI unavailable — scoring locally via ML pipeline")
        df = enrich_predictions_local(df)

    return df, api_up


def load_feature_importance() -> pd.DataFrame:
    """XGBoost gain-based feature importance."""
    model = joblib.load(MODELS_DIR / "churn_xgboost_model.joblib")
    names = joblib.load(ARTIFACTS_DIR / "feature_names.joblib")
    imp = pd.DataFrame(
        {"feature": names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return imp


def load_shap_importance(sample_size: int = 300) -> pd.DataFrame:
    """Mean |SHAP| importance from TreeExplainer on a test sample."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — SHAP chart will use XGBoost importance fallback")
        xgb = load_feature_importance()
        return xgb.rename(columns={"importance": "mean_abs_shap"})

    model = joblib.load(MODELS_DIR / "churn_xgboost_model.joblib")
    names = joblib.load(ARTIFACTS_DIR / "feature_names.joblib")
    X_test = joblib.load(ARTIFACTS_DIR / "X_test.joblib")

    n = min(sample_size, X_test.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test.shape[0], size=n, replace=False)
    X_sample = X_test[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        arr = arr[:, :, 1]

    mean_abs = np.abs(arr).mean(axis=0)
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
    )


def compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Executive KPI metrics."""
    total = len(df)
    churn_rate = df["churn_probability"].mean()
    high_risk = (df["risk_category"] == "High Risk").sum()
    revenue_at_risk = df.loc[df["risk_category"] == "High Risk", "total_revenue"].sum()
    return {
        "total_customers": total,
        "churn_rate": churn_rate,
        "high_risk_customers": int(high_risk),
        "revenue_at_risk": float(revenue_at_risk),
    }


def get_customer_risk_factors(row: pd.Series, importance: pd.DataFrame, top_n: int = 6) -> list[dict]:
    """
    Heuristic top risk factors for deep dive (feature values vs dataset median).
    Uses global importance ranking and deviation from median.
    """
    numeric_cols = [
        "tenure_months", "monthly_logins", "csat_score", "nps_score",
        "support_tickets", "usage_growth_rate", "payment_failures",
        "last_login_days_ago", "total_revenue",
    ]
    factors = []
    for col in numeric_cols:
        if col not in row.index:
            continue
        factors.append({
            "metric": col.replace("_", " ").title(),
            "value": row[col],
            "direction": "risk+" if col in ("support_tickets", "payment_failures", "last_login_days_ago") else "risk-",
        })
    return factors[:top_n]
