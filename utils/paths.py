"""
Central path definitions for reproducible file access across the project.
"""

from pathlib import Path

# Project root (parent of utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Default dataset filename for EDA and training pipelines
DEFAULT_DATASET_FILENAME = "customer_churn_business_dataset.csv"
DEFAULT_DATASET_PATH = DATA_DIR / DEFAULT_DATASET_FILENAME
