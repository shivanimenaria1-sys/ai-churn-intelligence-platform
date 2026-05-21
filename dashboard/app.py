"""
B2B Churn Prediction — Executive Dashboard (Plotly Dash)

Run from project root:
    python dashboard/app.py

Prerequisites:
    1. Trained model artifacts in models/ and models/artifacts/
    2. Dataset at data/customer_churn_business_dataset.csv
    3. (Recommended) FastAPI backend for live /predict integration:
       uvicorn backend.main:app --reload

Open: http://127.0.0.1:8050
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path when running as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dash
import dash_bootstrap_components as dbc

from dashboard.callbacks import register_callbacks
from dashboard.config import DASHBOARD_HOST, DASHBOARD_PORT, API_BASE_URL
from dashboard.data_service import preload_dashboard_data
from dashboard.layout import build_layout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="B2B Churn Analytics | ChurnGuard",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

app.layout = build_layout()
register_callbacks(app)
server = app.server  # for gunicorn / production WSGI


def main() -> None:
    """Start the Dash development server."""
    port = int(os.environ.get("PORT", DASHBOARD_PORT))
    host = "0.0.0.0"
    
    logger.info("Preloading data before starting server…")
    preload_dashboard_data(use_api=False)
    logger.info("Starting ChurnGuard dashboard at http://%s:%s", host, port)
    logger.info("Using backend API: %s", API_BASE_URL)
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
