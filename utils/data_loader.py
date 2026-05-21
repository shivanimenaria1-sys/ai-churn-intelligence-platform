"""
Data loading helpers for churn datasets.

Used by notebooks and any batch scripts that read the canonical CSV
from ``data/customer_churn_business_dataset.csv``.
"""

from pathlib import Path

import pandas as pd

from utils.paths import DEFAULT_DATASET_PATH


def load_churn_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the B2B customer churn CSV into a DataFrame.

    Parameters
    ----------
    path : Path | str | None
        Optional override for dataset location. Defaults to
        data/customer_churn_business_dataset.csv under project root.

    Returns
    -------
    pd.DataFrame
        Raw churn dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the resolved path.
    """
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Place customer_churn_business_dataset.csv in the data/ directory."
        )

    return pd.read_csv(dataset_path)
