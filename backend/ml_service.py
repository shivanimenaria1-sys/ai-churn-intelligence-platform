"""
Model artifact loading and churn inference service.
"""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.config import (
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    REFERENCE_DATA_PATH,
    RISK_THRESHOLDS,
)
from backend.feature_builder import (
    ReferenceStats,
    build_feature_row,
    get_preprocessor_input_columns,
)

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is requested before artifacts are loaded."""


class ChurnModelService:
    """Loads artifacts once and serves churn predictions."""

    def __init__(self) -> None:
        self.model: Any = None
        self.preprocessor: Any = None
        self.feature_names: np.ndarray | None = None
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.ref_stats: ReferenceStats | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load_artifacts(self) -> None:
        """Load model, preprocessor, and feature names from disk."""
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        if not PREPROCESSOR_PATH.is_file():
            raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
        if not FEATURE_NAMES_PATH.is_file():
            raise FileNotFoundError(f"Feature names not found: {FEATURE_NAMES_PATH}")

        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        self.feature_names = joblib.load(FEATURE_NAMES_PATH)

        self.numeric_columns, self.categorical_columns = get_preprocessor_input_columns(
            self.preprocessor
        )

        if REFERENCE_DATA_PATH.is_file():
            self.ref_stats = ReferenceStats.from_csv(REFERENCE_DATA_PATH)
        else:
            self.ref_stats = None
            logger.warning(
                "Reference data CSV not found at %s; engineered features use neutral scaling.",
                REFERENCE_DATA_PATH,
            )

        self._loaded = True
        logger.info("Model loaded successfully from %s", MODEL_PATH)
        logger.info("Preprocessor loaded successfully from %s", PREPROCESSOR_PATH)
        logger.info(
            "Feature count: %d encoded | %d raw inputs (%d numeric + %d categorical)",
            len(self.feature_names),
            len(self.numeric_columns) + len(self.categorical_columns),
            len(self.numeric_columns),
            len(self.categorical_columns),
        )

    @staticmethod
    def assign_risk_category(probability: float) -> str:
        if probability < RISK_THRESHOLDS["low"]:
            return "Low Risk"
        if probability < RISK_THRESHOLDS["high"]:
            return "Medium Risk"
        return "High Risk"

    def predict(self, customer: dict) -> dict:
        """
        Run preprocessing and model inference for one customer record.

        Returns dict with churn_probability, predicted_class, risk_category,
        and confidence_score.
        """
        if not self._loaded:
            raise ModelNotLoadedError("Artifacts not loaded. Call load_artifacts() at startup.")

        try:
            features_df = build_feature_row(
                payload=customer,
                numeric_columns=self.numeric_columns,
                categorical_columns=self.categorical_columns,
                ref_stats=self.ref_stats,
            )
            processed = self.preprocessor.transform(features_df)

            if processed.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Preprocessor output dimension {processed.shape[1]} "
                    f"!= expected {len(self.feature_names)}"
                )

            proba = float(self.model.predict_proba(processed)[0, 1])
            predicted_class = int(proba >= 0.5)
            confidence = float(max(proba, 1.0 - proba))

            return {
                "churn_probability": round(proba, 6),
                "predicted_class": predicted_class,
                "risk_category": self.assign_risk_category(proba),
                "confidence_score": round(confidence, 6),
            }

        except ValueError as exc:
            logger.exception("Feature building or preprocessing failed")
            raise ValueError(f"Preprocessing failed: {exc}") from exc
        except Exception as exc:
            logger.exception("Model prediction failed")
            raise RuntimeError(f"Prediction failed: {exc}") from exc

    def get_model_info(self) -> dict:
        if not self._loaded:
            raise ModelNotLoadedError("Artifacts not loaded.")

        model_type = type(self.model).__name__
        return {
            "model_type": model_type,
            "feature_count": int(len(self.feature_names)),
            "input_feature_count": len(self.numeric_columns) + len(self.categorical_columns),
            "risk_thresholds": RISK_THRESHOLDS,
            "encoded_feature_names_sample": list(self.feature_names[:10]),
            "training_summary": (
                f"{model_type} binary classifier with {len(self.feature_names)} "
                "encoded features (StandardScaler + OneHotEncoder pipeline)."
            ),
        }


# Singleton used by FastAPI routes
churn_service = ChurnModelService()
