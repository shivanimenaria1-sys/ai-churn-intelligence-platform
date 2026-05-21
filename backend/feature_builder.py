"""
Build model-input features from raw API payloads.

Mirrors feature_engineering.ipynb logic: composite scores, log transforms,
and column ordering expected by the fitted sklearn preprocessor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns created via log1p in training (inferred from preprocessor)
LOG_SOURCE_COLUMNS = [
    "tenure_months",
    "payment_failures",
    "avg_resolution_time",
    "escalations",
    "last_login_days_ago",
    "total_revenue",
]

ENGINEERED_COLUMNS = [
    "engagement_score",
    "support_risk_score",
    "inactivity_risk",
    "satisfaction_risk",
    "customer_value_score",
    "payment_risk_score",
    "marketing_engagement_score",
]


class ReferenceStats:
    """Median/IQR statistics from training data for robust feature scaling."""

    def __init__(self, medians: dict[str, float], iqrs: dict[str, float]) -> None:
        self.medians = medians
        self.iqrs = iqrs

    @classmethod
    def from_csv(cls, csv_path: Path) -> ReferenceStats:
        df = pd.read_csv(csv_path)
        numeric = df.select_dtypes(include=[np.number])
        medians = numeric.median().to_dict()
        iqrs = (numeric.quantile(0.75) - numeric.quantile(0.25)).to_dict()
        logger.info("Reference stats loaded from %s (%d numeric columns)", csv_path, len(medians))
        return cls(medians=medians, iqrs=iqrs)

    def robust(self, value: float, column: str) -> float:
        iqr = self.iqrs.get(column, 0.0)
        if iqr == 0:
            return 0.0
        return (value - self.medians.get(column, value)) / iqr


def get_preprocessor_input_columns(preprocessor_pipeline) -> tuple[list[str], list[str]]:
    """
    Extract numeric and categorical input column names from a fitted
    ColumnTransformer inside the churn preprocessor pipeline.
    """
    column_transformer = preprocessor_pipeline.named_steps["preprocessor"]
    numeric_cols = list(column_transformer.transformers_[0][2])
    categorical_cols = list(column_transformer.transformers_[1][2])
    return numeric_cols, categorical_cols


def build_feature_row(
    payload: dict,
    numeric_columns: list[str],
    categorical_columns: list[str],
    ref_stats: ReferenceStats | None,
) -> pd.DataFrame:
    """
    Transform raw customer payload into a single-row DataFrame matching
    training-time columns (including engineered and log features).
    """
    row = dict(payload)

    if ref_stats is None:
        logger.warning(
            "Reference stats unavailable; engineered scores default to 0.0. "
            "Place training CSV at data/customer_churn_business_dataset.csv."
        )
        ref = ReferenceStats(medians={}, iqrs={})
    else:
        ref = ref_stats

    # Composite business features (same formulas as feature_engineering.ipynb)
    row["engagement_score"] = (
        ref.robust(row["monthly_logins"], "monthly_logins")
        + ref.robust(row["weekly_active_days"], "weekly_active_days")
        + ref.robust(row["avg_session_time"], "avg_session_time")
        + ref.robust(row["features_used"], "features_used")
    ) / 4

    row["support_risk_score"] = (
        ref.robust(row["support_tickets"], "support_tickets")
        + ref.robust(row["avg_resolution_time"], "avg_resolution_time")
        + ref.robust(row["escalations"], "escalations")
    ) / 3

    row["inactivity_risk"] = ref.robust(row["last_login_days_ago"], "last_login_days_ago")

    row["satisfaction_risk"] = (
        -ref.robust(row["csat_score"], "csat_score")
        + -ref.robust(row["nps_score"], "nps_score")
    ) / 2

    row["customer_value_score"] = (
        ref.robust(row["total_revenue"], "total_revenue")
        + ref.robust(row["tenure_months"], "tenure_months")
    ) / 2

    row["payment_risk_score"] = (
        ref.robust(row["payment_failures"], "payment_failures")
        + ref.robust(row["price_increase_last_3m"], "price_increase_last_3m")
    ) / 2

    row["marketing_engagement_score"] = (
        ref.robust(row["email_open_rate"], "email_open_rate")
        + ref.robust(row["marketing_click_rate"], "marketing_click_rate")
    ) / 2

    # Log transforms for skewed numerics used in training
    for col in LOG_SOURCE_COLUMNS:
        if col in row:
            row[f"log_{col}"] = float(np.log1p(max(row[col], 0)))

    # Assemble in exact column order expected by ColumnTransformer
    ordered_columns = numeric_columns + categorical_columns
    missing = [c for c in ordered_columns if c not in row]
    if missing:
        raise ValueError(f"Missing required feature fields after engineering: {missing}")

    return pd.DataFrame([{col: row[col] for col in ordered_columns}], columns=ordered_columns)
