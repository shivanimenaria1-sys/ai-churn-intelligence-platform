#!/usr/bin/env python
"""
Verify project imports and required artifacts (run from project root).

Usage:
    python scripts/verify_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_imports() -> list[str]:
    """Import core packages and application modules."""
    errors: list[str] = []
    modules = [
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "fastapi",
        "dash",
        "plotly",
        "backend.config",
        "backend.ml_service",
        "backend.feature_builder",
        "dashboard.config",
        "dashboard.data_service",
        "utils.paths",
        "utils.data_loader",
    ]
    for name in modules:
        try:
            __import__(name)
            print(f"  OK  {name}")
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
            print(f"  FAIL {name}: {exc}")
    return errors


def check_artifacts() -> list[str]:
    """Confirm model files exist for API/dashboard."""
    missing: list[str] = []
    required = [
        PROJECT_ROOT / "models" / "churn_xgboost_model.joblib",
        PROJECT_ROOT / "models" / "artifacts" / "churn_preprocessor.joblib",
        PROJECT_ROOT / "models" / "artifacts" / "feature_names.joblib",
    ]
    for path in required:
        if path.is_file():
            print(f"  OK  {path.relative_to(PROJECT_ROOT)}")
        else:
            missing.append(str(path.relative_to(PROJECT_ROOT)))
            print(f"  MISSING  {path.relative_to(PROJECT_ROOT)}")
    return missing


def check_dataset() -> bool:
    path = PROJECT_ROOT / "data" / "customer_churn_business_dataset.csv"
    ok = path.is_file()
    print(f"  {'OK' if ok else 'MISSING'}  {path.relative_to(PROJECT_ROOT)}")
    return ok


def main() -> int:
    print("=== Import check ===")
    import_errors = check_imports()
    print("\n=== Artifact check ===")
    missing_artifacts = check_artifacts()
    print("\n=== Dataset check ===")
    has_data = check_dataset()

    if import_errors:
        print("\nInstall dependencies: pip install -r requirements.txt")
        return 1
    if missing_artifacts:
        print("\nRun notebooks to generate models (see models/README.md).")
        return 1
    if not has_data:
        print("\nAdd dataset to data/ (see data/README.md).")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
