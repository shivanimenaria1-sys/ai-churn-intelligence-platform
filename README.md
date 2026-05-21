# B2B Customer Churn Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![Dash](https://img.shields.io/badge/Dash-2.14%2B-purple.svg)](https://dash.plotly.com/)

End-to-end **B2B churn prediction** platform: exploratory analysis, feature engineering, XGBoost modeling, SHAP explainability, a production **FastAPI** scoring API, and an executive **Plotly Dash** dashboard.

---

## Highlights (portfolio)

- **ML pipeline:** EDA → feature engineering → stratified train/test → XGBoost with class imbalance handling
- **Explainable AI:** SHAP summary, global importance, and waterfall plots for account-level narratives
- **Production API:** FastAPI with Pydantic validation, risk tiers, and health checks
- **Executive dashboard:** KPIs, risk segmentation, interactive filters, live API integration
- **Modular codebase:** Separated `backend/`, `dashboard/`, `utils/`, and `notebooks/`

---

## Architecture

```
data/ ──► notebooks/ ──► models/
              │              │
              ▼              ▼
           utils/ ◄──── backend/ (FastAPI)
              │
              ▼
          dashboard/ (Plotly Dash)
```

| Component | Stack | Role |
|-----------|-------|------|
| Notebooks | Jupyter, Pandas, Seaborn | EDA, training, SHAP |
| Models | XGBoost, scikit-learn, joblib | Classifier + preprocessor |
| Backend | FastAPI, Uvicorn | Real-time `/predict` |
| Dashboard | Dash, Plotly, Bootstrap | BI & retention ops |

---

## Project structure

```
b2b-churn-predictor/
├── backend/                 # FastAPI REST API
│   ├── main.py
│   ├── ml_service.py
│   ├── feature_builder.py
│   ├── schemas.py
│   └── config.py
├── dashboard/               # Executive Dash app
│   ├── app.py
│   ├── callbacks.py
│   ├── data_service.py
│   └── assets/
├── data/                    # Dataset (gitignored CSV)
├── models/                  # Trained artifacts (gitignored)
├── notebooks/               # ML workflow notebooks
├── utils/                   # Shared paths & data loaders
├── scripts/                 # Setup verification
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd b2b-churn-predictor
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add data

Place `customer_churn_business_dataset.csv` in `data/` (see [data/README.md](data/README.md)).

### 3. Train models

```bash
jupyter notebook notebooks/feature_engineering.ipynb
jupyter notebook notebooks/model_training.ipynb
```

See [models/README.md](models/README.md) for artifact details.

### 4. Verify setup

```bash
python scripts/verify_setup.py
```

### 5. Run API

```bash
uvicorn backend.main:app --reload --port 8001
```

- Docs: http://127.0.0.1:8001/docs  
- Health: http://127.0.0.1:8001/health  

### 6. Run dashboard

```bash
python dashboard/app.py
```

- UI: http://127.0.0.1:8050  

Configure API URL in `dashboard/config.py` if not using port `8001`.

---

## API example

```bash
curl -X POST "http://127.0.0.1:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:** `churn_probability`, `predicted_class`, `risk_category`, `confidence_score`

---

## Risk tiers

| Tier | Probability |
|------|-------------|
| Low Risk | &lt; 0.33 |
| Medium Risk | 0.33 – 0.66 |
| High Risk | ≥ 0.66 |

---

## Tech stack

Python 3.10+ · Pandas · scikit-learn · XGBoost · SHAP · FastAPI · Plotly Dash · Jupyter

---

## License

[MIT](LICENSE)
