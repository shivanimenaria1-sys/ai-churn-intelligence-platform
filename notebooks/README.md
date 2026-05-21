# Notebooks

End-to-end ML workflow for B2B churn prediction. Run sequentially:

| Notebook | Purpose |
|----------|---------|
| `eda.ipynb` | Exploratory data analysis and business insights |
| `feature_engineering.ipynb` | Cleaning, feature engineering, preprocessing pipeline |
| `model_training.ipynb` | XGBoost training, evaluation, SHAP explainability |

**Prerequisite:** Dataset at `../data/customer_churn_business_dataset.csv`

Outputs are saved to `../models/` and consumed by the FastAPI backend and Dash dashboard.
