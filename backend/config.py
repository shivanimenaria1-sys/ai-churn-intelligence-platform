"""
Application configuration and artifact paths.
"""

from pathlib import Path

# Project root (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODELS_DIR / "churn_xgboost_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "churn_preprocessor.joblib"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.joblib"
REFERENCE_DATA_PATH = DATA_DIR / "customer_churn_business_dataset.csv"

# Risk tier thresholds (aligned with model_training notebook)
RISK_THRESHOLDS = {"low": 0.33, "high": 0.66}

# API metadata
API_TITLE = "B2B Churn Predictor API"
API_DESCRIPTION = (
    "Real-time B2B customer churn risk scoring using a trained XGBoost model "
    "and sklearn preprocessing pipeline."
)
API_VERSION = "1.0.0"
