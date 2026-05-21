# Data

Place the B2B churn dataset here:

```
customer_churn_business_dataset.csv
```

## Expected columns

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | string | Unique account identifier |
| `tenure_months` | int | Months as a customer |
| `monthly_logins` | int | Monthly product logins |
| `avg_session_time` | float | Average session duration |
| `usage_growth_rate` | float | Usage growth rate |
| `support_tickets` | int | Support ticket count |
| `csat_score` | float | Customer satisfaction (1–10) |
| `nps_score` | float | Net Promoter Score |
| `payment_failures` | int | Failed payment count |
| `contract_type` | string | Monthly / Annual / Multi-year |
| `customer_segment` | string | SMB / Mid-Market / Enterprise |
| `weekly_active_days` | int | Active days per week |
| `features_used` | int | Product features adopted |
| `avg_resolution_time` | float | Avg ticket resolution (hours) |
| `escalations` | int | Escalated tickets |
| `last_login_days_ago` | float | Days since last login |
| `total_revenue` | float | Account revenue |
| `price_increase_last_3m` | int | 1 if price increased recently |
| `email_open_rate` | float | Email open rate (0–1) |
| `marketing_click_rate` | float | Marketing CTR (0–1) |
| `churn` | int | Target: 1 = churned, 0 = retained |

Raw CSV files are **gitignored** to keep the repository lightweight. Clone the repo and add your dataset locally before running notebooks or the dashboard.
