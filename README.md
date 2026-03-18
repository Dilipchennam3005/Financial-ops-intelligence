# Financial Operations Intelligence Platform
### Pinnacle Asset Management | Azure Data Factory · MySQL · Python · Power BI

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Azure](https://img.shields.io/badge/Azure-Data%20Factory-0078D4?logo=microsoftazure)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?logo=mysql)
![PowerBI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-brightgreen)
![Tests](https://img.shields.io/badge/Tests-38%20Passing-brightgreen)

---

## Overview

Built a financial operations intelligence platform for **Pinnacle Asset Management** (fictional) to automate fund performance reporting, detect NAV anomalies, forecast AUM cash flows, and segment funds by operational profile. The system processes 50 funds across 5 asset classes with $53B in AUM across 65,250 daily NAV records (2020–2024).

---

## Key Results

| Metric | Value |
|--------|-------|
| Funds monitored | 50 across 5 asset classes |
| Total AUM | ~$53B |
| NAV records | 65,250 (2020–2024) |
| Cash flow transactions | 16,572 |
| Trade records | 24,484 |
| Prophet forecast MAPE | 1.71% |
| 90-day AUM forecast | $81.45B → $85.83B |
| NAV anomalies flagged | 651 (1.0% rate) |
| Fund segments | 5 operational profiles |
| Test coverage | 38 tests passing |

---

## Architecture

```
Data Generation (Python)
        │
        ▼
Azure Blob Storage ──► Azure Data Factory Pipeline
        │                      │
        │              Daily trigger (6AM UTC)
        │                      │
        ▼                      ▼
    MySQL pinnacle_ops database
    ├── fund_master
    ├── nav_daily
    ├── cash_flows
    ├── trades
    └── monthly_performance
              │
              ▼
    Python ML Pipeline
    ├── Prophet — 90-day AUM forecasting
    ├── Random Forest + Isolation Forest — NAV anomaly detection
    └── K-Means — Fund segmentation (5 profiles)
              │
              ▼
    Power BI Executive Dashboard
    ├── Page 1: Portfolio Overview
    ├── Page 2: Fund Performance
    └── Page 3: Risk & Operations
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud Orchestration | Azure Data Factory (pipelines, linked services, ARM templates) |
| Database | MySQL 8.0 (5 tables, 4 analytical views) |
| ML — Forecasting | Prophet (time-series, COVID + rate hike regressors) |
| ML — Anomaly Detection | Random Forest + Isolation Forest ensemble |
| ML — Segmentation | K-Means clustering (5 fund profiles) |
| Visualization | Power BI Desktop (3-page dashboard, 5 DAX measures) |
| Languages | Python, SQL, DAX |
| Testing | pytest (38 tests) |
| DevOps | Git, GitHub |

---

## Repository Structure

```
financial-ops-intelligence/
├── data/
│   ├── generate_fund_data.py          # Synthetic data generator
│   ├── raw/                           # Generated CSVs (gitignored)
│   └── processed/                     # ML output files
├── azure/
│   ├── adf_pipelines/
│   │   ├── pipelines/
│   │   │   └── PL_FundDataIngestion.json
│   │   ├── linked_services/
│   │   │   ├── LS_AzureBlobStorage.json
│   │   │   └── LS_MySQL_PinnacleOps.json
│   │   └── triggers/
│   │       └── TR_DailyFundIngestion.json
│   └── arm_templates/
│       └── adf_deployment.json
├── mysql/
│   └── 01_schema.sql                  # DDL: 5 tables, 4 views
├── ml/
│   └── ml_pipeline.py                 # Prophet + RF + K-Means
├── powerbi/
│   └── pinnacle_ops_dashboard.pbix    # 3-page Power BI dashboard
├── tests/
│   └── unit/
│       └── test_pipeline.py           # 38 tests
└── README.md
```

---

## Dataset

- **50 funds** across Equity, Fixed Income, Multi-Asset, Alternative, Money Market
- **65,250 daily NAV records** (2020–2024, business days only)
- **16,572 cash flow transactions** (subscriptions, redemptions, fees, dividends)
- **24,484 trade records** across 16 securities
- **Economic regimes:** COVID shock (2020), Fed rate hike cycle (2022–2023)
- **666 injected NAV anomalies** for model training

---

## ML Pipeline

### Model 1: Prophet — 90-Day AUM Forecasting
- Trained on 2020–2023 daily AUM data
- Custom regressors: `covid_period`, `rate_hike_period`
- Multiplicative seasonality (yearly + weekly)
- **Test MAPE: 1.71%** — production-grade accuracy
- 90-day forecast: $81.45B → $85.83B (confidence: $81.57B–$89.45B)

### Model 2: Random Forest + Isolation Forest — NAV Anomaly Detection
- 10 engineered features: rolling return std, z-score, benchmark deviation
- Two-stage ensemble: Isolation Forest (unsupervised) + RF (supervised)
- Flagged 651 anomalies (1.0% rate) for fund accounting review
- Top features: `return_20d_std`, `return_5d_std`, `return_z_score`

### Model 3: K-Means — Fund Segmentation
- 9 features: avg return, volatility, Sharpe ratio, AUM, anomaly count
- **5 operational profiles:**

| Profile | Funds | Avg Return | Avg Sharpe | Avg AUM |
|---------|-------|-----------|-----------|---------|
| Large Cap Stable | 5 | 1.04% | 0.539 | $5.5B |
| High Operational Risk | 11 | 1.37% | 0.102 | $1.5B |
| High Growth | 7 | 1.26% | 0.063 | $422M |
| Cluster 0 | 18 | 0.66% | 0.075 | $915M |
| Underperformers | 9 | -0.03% | -0.002 | $724M |

---

## Azure Data Factory Pipeline

- **PL_FundDataIngestion:** Master pipeline with 6 activities
  - Copy fund_master → MySQL (with truncate)
  - Copy nav_daily → MySQL (5,000 batch size)
  - Copy cash_flows → MySQL
  - Copy trades → MySQL
  - Validate row counts (Lookup activity)
  - Log pipeline run (Stored Procedure)
- **TR_DailyFundIngestion:** Scheduled trigger at 6:00 AM UTC daily
- **ARM template:** Full infrastructure-as-code deployment

---

## MySQL Schema

```sql
pinnacle_ops
├── fund_master          -- 50 fund records
├── nav_daily            -- 65,250 daily NAV records
├── cash_flows           -- 16,572 transactions
├── trades               -- 24,484 trade records
├── monthly_performance  -- 3,000 monthly summaries
└── ml_scores            -- Model output scores
```

4 analytical views: `vw_aum_by_asset_class`, `vw_fund_performance`,
`vw_cashflow_summary`, `vw_nav_anomalies`

---

## Power BI Dashboard

**Page 1 — Portfolio Overview**
- KPI cards: Total AUM, Total Funds, Avg Sharpe Ratio, Total Anomalies
- Donut chart: AUM distribution by asset class
- Line chart: Monthly return trend by asset class (2020–2024)
- Asset class slicer

**Page 2 — Fund Performance**
- Bar chart: Active return ranking by fund
- Scatter plot: Risk vs Return by fund segment (K-Means clusters)
- Table: Fund leaderboard with Sharpe ratio and cluster label

**Page 3 — Risk & Operations**
- Line chart: NAV anomalies over time
- Stacked bar: Cash flow by transaction type and status
- Donut: Transaction status breakdown (Settled/Pending/Failed)

---

## Running the Project

```bash
# 1. Generate data
python data/generate_fund_data.py

# 2. Set up MySQL
mysql -u root -p < mysql/01_schema.sql

# 3. Run ML pipeline
pip install prophet scikit-learn
python ml/ml_pipeline.py

# 4. Run tests
pytest tests/ -v
```

---

## Author

**Dilip Chennam** | Data Analyst | [GitHub](https://github.com/Dilipchennam3005)
