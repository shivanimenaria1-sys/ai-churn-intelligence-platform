"""
Pydantic request/response schemas for the churn prediction API.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """
    Raw customer account features for churn scoring.

    Provide all fields below; engineered and log features are derived server-side
    before the sklearn preprocessor runs.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "customer_segment": "Mid-Market",
            }
        }
    )

    # Core product & relationship metrics
    tenure_months: int = Field(..., ge=0, description="Months as a paying customer")
    monthly_logins: int = Field(..., ge=0, description="Logins in the last month")
    avg_session_time: float = Field(..., ge=0, description="Average session duration (minutes)")
    usage_growth_rate: float = Field(..., description="Month-over-month usage growth rate")
    support_tickets: int = Field(..., ge=0, description="Support tickets in the period")
    csat_score: float = Field(..., ge=1, le=10, description="Customer satisfaction score")
    nps_score: float = Field(..., ge=-100, le=100, description="Net Promoter Score")
    payment_failures: int = Field(..., ge=0, description="Failed payment attempts")

    # Extended engagement & operations
    weekly_active_days: int = Field(..., ge=0, le=7, description="Active days per week")
    features_used: int = Field(..., ge=0, description="Distinct product features used")
    avg_resolution_time: float = Field(..., ge=0, description="Avg ticket resolution time (hours)")
    escalations: int = Field(..., ge=0, description="Escalated support cases")
    last_login_days_ago: float = Field(..., ge=0, description="Days since last login")
    total_revenue: float = Field(..., ge=0, description="Account revenue (USD)")
    price_increase_last_3m: int = Field(..., ge=0, le=1, description="1 if price increased in last 3 months")
    email_open_rate: float = Field(..., ge=0, le=1, description="Marketing email open rate")
    marketing_click_rate: float = Field(..., ge=0, le=1, description="Marketing click-through rate")

    # Commercial attributes
    contract_type: Literal["Monthly", "Annual", "Multi-year"] = Field(
        ..., description="Contract billing term"
    )
    customer_segment: Literal["SMB", "Mid-Market", "Enterprise"] = Field(
        ..., description="Customer segment tier"
    )


class PredictionResponse(BaseModel):
    """Churn prediction output for a single customer."""

    churn_probability: float = Field(..., ge=0, le=1, description="P(churn=1)")
    predicted_class: int = Field(..., description="0=retained, 1=churn")
    risk_category: str = Field(..., description="Low Risk | Medium Risk | High Risk")
    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in predicted class: max(p, 1-p)",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool


class ModelInfoResponse(BaseModel):
    model_type: str
    feature_count: int
    training_summary: str
    risk_thresholds: dict[str, float]
    input_feature_count: int
    encoded_feature_names_sample: list[str]
