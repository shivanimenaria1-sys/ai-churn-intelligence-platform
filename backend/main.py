"""
FastAPI application for real-time B2B churn prediction.

Run locally:
    uvicorn backend.main:app --reload

Swagger UI (sample payload pre-filled):
    http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import API_DESCRIPTION, API_TITLE, API_VERSION
from backend.ml_service import ModelNotLoadedError, churn_service
from backend.schemas import (
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — load artifacts at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts when the API starts."""
    try:
        churn_service.load_artifacts()
    except FileNotFoundError as exc:
        logger.error("Startup failed — missing artifact: %s", exc)
        raise
    except Exception as exc:
        logger.error("Startup failed — could not load artifacts: %s", exc)
        raise
    yield
    logger.info("Shutting down churn API")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return clear 422 messages for invalid or missing request fields."""
    errors = []
    for err in exc.errors():
        loc = " → ".join(str(part) for part in err.get("loc", []))
        errors.append({"field": loc, "message": err.get("msg", "Invalid value")})
    logger.warning("Validation error on %s: %s", request.url.path, errors)
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input payload", "errors": errors},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """API status and documentation links."""
    return {
        "message": "B2B Churn Predictor API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "model_info": "/model-info",
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check() -> HealthResponse:
    """Liveness probe for orchestrators and load balancers."""
    return HealthResponse(
        status="healthy" if churn_service.is_loaded else "degraded",
        model_loaded=churn_service.model is not None,
        preprocessor_loaded=churn_service.preprocessor is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
async def model_info() -> ModelInfoResponse:
    """Metadata about the loaded model and risk tier thresholds."""
    try:
        info = churn_service.get_model_info()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ModelInfoResponse(
        model_type=info["model_type"],
        feature_count=info["feature_count"],
        training_summary=info["training_summary"],
        risk_thresholds=info["risk_thresholds"],
        input_feature_count=info["input_feature_count"],
        encoded_feature_names_sample=info["encoded_feature_names_sample"],
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Score a single customer for churn risk",
    response_description="Churn probability, class, risk tier, and confidence",
)
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    """
    Score one B2B account for churn risk.

    **Sample JSON** (also available in Swagger "Example Value"):

    ```json
    {
      "tenure_months": 18,
      "monthly_logins": 28,
      "avg_session_time": 42.5,
      "usage_growth_rate": 0.08,
      "support_tickets": 2,
      "csat_score": 7.8,
      "nps_score": 42.0,
      "payment_failures": 0,
      "weekly_active_days": 4,
      "features_used": 12,
      "avg_resolution_time": 14.0,
      "escalations": 0,
      "last_login_days_ago": 3.0,
      "total_revenue": 48000.0,
      "price_increase_last_3m": 0,
      "email_open_rate": 0.35,
      "marketing_click_rate": 0.12,
      "contract_type": "Annual",
      "customer_segment": "Mid-Market"
    }
    ```
    """
    if not churn_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    payload = customer.model_dump()

    try:
        result = churn_service.predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "Prediction — class=%s risk=%s proba=%.4f",
        result["predicted_class"],
        result["risk_category"],
        result["churn_probability"],
    )

    return PredictionResponse(**result)
