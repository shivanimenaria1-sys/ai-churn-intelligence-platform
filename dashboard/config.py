"""
Dashboard configuration — paths, API settings, and visual theme.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_business_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# Use Render backend by default, allow override for local backend testing
API_BASE_URL = os.environ.get("API_BASE_URL", "https://ai-churn-intelligence-platform.onrender.com")
API_PREDICT_URL = f"{API_BASE_URL}/predict"
API_HEALTH_URL = f"{API_BASE_URL}/health"

# Risk thresholds (aligned with backend)
RISK_LOW = 0.33
RISK_HIGH = 0.66

RISK_COLORS = {
    "Low Risk": "#28a745",
    "Medium Risk": "#ffc107",
    "High Risk": "#dc3545",
}

# Enterprise theme
THEME = {
    "primary": "#1e3a5f",
    "secondary": "#4a90d9",
    "background": "#f4f6f9",
    "card": "#ffffff",
    "text": "#2c3e50",
    "muted": "#6c757d",
}

# Columns sent to /predict (must match backend CustomerFeatures)
PREDICT_PAYLOAD_COLUMNS = [
    "tenure_months",
    "monthly_logins",
    "avg_session_time",
    "usage_growth_rate",
    "support_tickets",
    "csat_score",
    "nps_score",
    "payment_failures",
    "weekly_active_days",
    "features_used",
    "avg_resolution_time",
    "escalations",
    "last_login_days_ago",
    "total_revenue",
    "price_increase_last_3m",
    "email_open_rate",
    "marketing_click_rate",
    "contract_type",
    "customer_segment",
]

DASHBOARD_PORT = 8050
DASHBOARD_HOST = "127.0.0.1"

# Limit rows for faster startup (None = use full dataset)
MAX_CUSTOMERS: int | None = None
API_SCORE_SAMPLE = 300  # rows scored via live /predict when API is up
