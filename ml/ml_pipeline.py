"""
================================================================================
Pinnacle Asset Management — Financial Operations Intelligence Platform
ML Models: Forecasting, Anomaly Detection, Fund Segmentation
================================================================================
Models:
  1. Prophet       — 90-day AUM cash flow forecasting
  2. Random Forest — NAV anomaly detection
  3. K-Means       — Fund segmentation into operational profiles

Author: Dilip Chennam
================================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_score, recall_score, f1_score)
from prophet import Prophet

print("=" * 65)
print("Pinnacle Asset Management — ML Pipeline")
print("=" * 65)

# ── Load data
print("\nLoading data...")
funds    = pd.read_csv('data/raw/fund_master.csv')
nav      = pd.read_csv('data/raw/nav_daily.csv', parse_dates=['nav_date'])
cf       = pd.read_csv('data/raw/cash_flows.csv', parse_dates=['transaction_date'])
monthly  = pd.read_csv('data/raw/monthly_performance.csv')

print(f"  Funds:        {len(funds)}")
print(f"  NAV records:  {len(nav):,}")
print(f"  Cash flows:   {len(cf):,}")
print(f"  Monthly perf: {len(monthly):,}")

# ============================================================
# MODEL 1 — PROPHET: 90-Day AUM Cash Flow Forecasting
# ============================================================
print("\n" + "=" * 65)
print("MODEL 1: Prophet — 90-Day AUM Forecasting")
print("=" * 65)

# Aggregate daily AUM across all funds
daily_aum = nav.groupby('nav_date')['aum'].sum().reset_index()
daily_aum.columns = ['ds', 'y']
daily_aum = daily_aum.sort_values('ds')

# Train on 2020-2023, forecast 2024
train_aum = daily_aum[daily_aum['ds'] < '2024-01-01']
test_aum  = daily_aum[daily_aum['ds'] >= '2024-01-01']

# Build Prophet model with financial regressors
prophet_model = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10,
    holidays_prior_scale=10,
    seasonality_mode='multiplicative',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)

# Add COVID and rate hike regressors
train_aum = train_aum.copy()
train_aum['covid_period']     = ((train_aum['ds'] >= '2020-03-01') & (train_aum['ds'] <= '2020-12-31')).astype(int)
train_aum['rate_hike_period'] = ((train_aum['ds'] >= '2022-01-01') & (train_aum['ds'] <= '2023-12-31')).astype(int)

prophet_model.add_regressor('covid_period')
prophet_model.add_regressor('rate_hike_period')
prophet_model.fit(train_aum)

# Forecast 90 days
future = prophet_model.make_future_dataframe(periods=90, freq='B')
future['covid_period']     = 0
future['rate_hike_period'] = ((future['ds'] >= '2022-01-01') & (future['ds'] <= '2023-12-31')).astype(int)
forecast = prophet_model.predict(future)

# Evaluation on test set
test_forecast = forecast[forecast['ds'].isin(test_aum['ds'])][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
test_merged   = test_aum.merge(test_forecast, on='ds')
mape = np.mean(np.abs((test_merged['y'] - test_merged['yhat']) / test_merged['y'])) * 100
mae  = np.mean(np.abs(test_merged['y'] - test_merged['yhat']))

print(f"\n  Training period: 2020-01-01 to 2023-12-31")
print(f"  Forecast horizon: 90 business days")
print(f"  Test MAPE:  {mape:.2f}%")
print(f"  Test MAE:   ${mae/1e6:.1f}M")

# 90-day forward forecast
future_90 = forecast.tail(90)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
print(f"\n  90-Day Forecast Summary:")
print(f"  Start AUM:  ${future_90['yhat'].iloc[0]/1e9:.2f}B")
print(f"  End AUM:    ${future_90['yhat'].iloc[-1]/1e9:.2f}B")
print(f"  Range:      ${future_90['yhat_lower'].iloc[-1]/1e9:.2f}B — ${future_90['yhat_upper'].iloc[-1]/1e9:.2f}B")

# Save forecast
os.makedirs('data/processed', exist_ok=True)
future_90.to_csv('data/processed/aum_forecast_90d.csv', index=False)
print(f"  ✅ Forecast saved to data/processed/aum_forecast_90d.csv")

# ============================================================
# MODEL 2 — RANDOM FOREST: NAV Anomaly Detection
# ============================================================
print("\n" + "=" * 65)
print("MODEL 2: Random Forest — NAV Anomaly Detection")
print("=" * 65)

# Feature engineering for anomaly detection
nav_feat = nav.copy()
nav_feat = nav_feat.sort_values(['fund_id', 'nav_date'])

# Rolling features per fund
nav_feat['return_5d_avg']   = nav_feat.groupby('fund_id')['daily_return'].transform(lambda x: x.rolling(5,  min_periods=1).mean())
nav_feat['return_20d_avg']  = nav_feat.groupby('fund_id')['daily_return'].transform(lambda x: x.rolling(20, min_periods=1).mean())
nav_feat['return_5d_std']   = nav_feat.groupby('fund_id')['daily_return'].transform(lambda x: x.rolling(5,  min_periods=1).std().fillna(0))
nav_feat['return_20d_std']  = nav_feat.groupby('fund_id')['daily_return'].transform(lambda x: x.rolling(20, min_periods=1).std().fillna(0))
nav_feat['aum_5d_avg']      = nav_feat.groupby('fund_id')['aum'].transform(lambda x: x.rolling(5,  min_periods=1).mean())
nav_feat['aum_change_pct']  = nav_feat.groupby('fund_id')['aum'].pct_change().fillna(0)
nav_feat['return_vs_bench'] = nav_feat['daily_return'] - nav_feat['benchmark_return']
nav_feat['return_z_score']  = nav_feat.groupby('fund_id')['daily_return'].transform(
    lambda x: (x - x.rolling(20, min_periods=5).mean()) / (x.rolling(20, min_periods=5).std() + 1e-8)
)

FEATURES = [
    'daily_return', 'benchmark_return', 'active_return',
    'return_5d_avg', 'return_20d_avg', 'return_5d_std', 'return_20d_std',
    'aum_change_pct', 'return_vs_bench', 'return_z_score'
]

nav_feat = nav_feat.dropna(subset=FEATURES)
X = nav_feat[FEATURES]
y = nav_feat['is_anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Two-stage approach: Isolation Forest for unsupervised detection + RF for classification
# Stage 1: Isolation Forest
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(X_train)
iso_scores = iso.score_samples(X)
iso_preds  = (iso.predict(X) == -1).astype(int)

# Stage 2: Random Forest on labeled data
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=3,
    class_weight={0: 1, 1: 50},  # heavy weight on anomaly class
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_preds = (rf_probs >= 0.3).astype(int)  # lower threshold for recall

rf_auc  = roc_auc_score(y_test, rf_probs)
rf_prec = precision_score(y_test, rf_preds, zero_division=0)
rf_rec  = recall_score(y_test, rf_preds, zero_division=0)
rf_f1   = f1_score(y_test, rf_preds, zero_division=0)

# Ensemble: combine IF score + RF prob
nav_feat['iso_score']       = iso_scores
nav_feat['rf_anomaly_prob'] = rf_model.predict_proba(X)[:, 1]
nav_feat['ensemble_score']  = (nav_feat['rf_anomaly_prob'] * 0.6 +
                                (1 + nav_feat['iso_score']) * 0.4)
nav_feat['rf_is_anomaly']   = (nav_feat['ensemble_score'] > nav_feat['ensemble_score'].quantile(0.99)).astype(int)

print(f"\n  Training samples: {len(X_train):,}")
print(f"  Test samples:     {len(X_test):,}")
print(f"  Anomaly rate:     {y.mean():.2%}")
print(f"\n  Random Forest AUC-ROC:  {rf_auc:.4f}")
print(f"  Precision:              {rf_prec:.4f}")
print(f"  Recall:                 {rf_rec:.4f}")
print(f"  F1 Score:               {rf_f1:.4f}")
print(f"  Ensemble flagged:       {nav_feat['rf_is_anomaly'].sum():,} anomalies ({nav_feat['rf_is_anomaly'].mean():.2%})")

# Feature importance
fi = pd.DataFrame({'feature': FEATURES, 'importance': rf_model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
print(f"\n  Top 5 Features:")
print(fi.head(5).to_string(index=False))

nav_feat[['fund_id', 'nav_date', 'rf_anomaly_prob', 'iso_score', 'ensemble_score', 'rf_is_anomaly']].to_csv(
    'data/processed/nav_anomaly_scores.csv', index=False)
print(f"\n  ✅ Anomaly scores saved to data/processed/nav_anomaly_scores.csv")

# ============================================================
# MODEL 3 — K-MEANS: Fund Segmentation
# ============================================================
print("\n" + "=" * 65)
print("MODEL 3: K-Means — Fund Segmentation")
print("=" * 65)

# Build fund-level feature matrix from monthly performance
fund_features = monthly.groupby('fund_id').agg(
    avg_return      = ('monthly_return',   'mean'),
    return_vol      = ('monthly_return',   'std'),
    avg_active_ret  = ('active_return',    'mean'),
    avg_sharpe      = ('sharpe_ratio',     'mean'),
    avg_aum         = ('avg_aum',          'mean'),
    total_anomalies = ('anomaly_count',    'sum'),
    consistency     = ('monthly_return',   lambda x: (x > 0).mean()),
).reset_index().fillna(0)

# Merge with fund master
fund_features = fund_features.merge(
    funds[['fund_id', 'asset_class', 'management_fee', 'target_volatility']], on='fund_id')

CLUSTER_FEATURES = [
    'avg_return', 'return_vol', 'avg_active_ret', 'avg_sharpe',
    'avg_aum', 'total_anomalies', 'consistency', 'management_fee', 'target_volatility'
]

scaler        = StandardScaler()
X_cluster     = scaler.fit_transform(fund_features[CLUSTER_FEATURES])

# Find optimal K using inertia
inertias = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

# Use K=5 (5 operational profiles)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
fund_features['cluster_id'] = kmeans.fit_predict(X_cluster)

# Label clusters based on characteristics
cluster_profiles = fund_features.groupby('cluster_id').agg(
    fund_count   = ('fund_id',         'count'),
    avg_return   = ('avg_return',      'mean'),
    avg_vol      = ('return_vol',      'mean'),
    avg_sharpe   = ('avg_sharpe',      'mean'),
    avg_aum_M    = ('avg_aum',         lambda x: x.mean()/1e6),
    avg_anomalies= ('total_anomalies', 'mean'),
    consistency  = ('consistency',     'mean'),
).round(4)

cluster_labels = {
    cluster_profiles['avg_return'].idxmax():  'High Growth',
    cluster_profiles['avg_sharpe'].idxmax():  'Risk-Adjusted Leaders',
    cluster_profiles['avg_aum_M'].idxmax():   'Large Cap Stable',
    cluster_profiles['avg_anomalies'].idxmax():'High Operational Risk',
    cluster_profiles['avg_return'].idxmin():  'Underperformers',
}
# Fill any unmapped clusters
for i in range(5):
    if i not in cluster_labels:
        cluster_labels[i] = f'Cluster {i}'

fund_features['cluster_label'] = fund_features['cluster_id'].map(cluster_labels)

print(f"\n  Funds segmented into 5 operational profiles:")
print(f"\n  {'Cluster':<25} {'Funds':>6} {'Avg Return':>12} {'Avg Sharpe':>12} {'Avg AUM($M)':>12} {'Anomalies':>10}")
print("  " + "-" * 80)
for cid, label in cluster_labels.items():
    row = cluster_profiles.loc[cid]
    print(f"  {label:<25} {int(row['fund_count']):>6} {row['avg_return']*100:>11.2f}% {row['avg_sharpe']:>12.4f} {row['avg_aum_M']:>11.1f}M {row['avg_anomalies']:>10.1f}")

fund_features.to_csv('data/processed/fund_segments.csv', index=False)
print(f"\n  ✅ Fund segments saved to data/processed/fund_segments.csv")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("ML PIPELINE SUMMARY")
print("=" * 65)
print(f"\n  Prophet AUM Forecasting:")
print(f"    Test MAPE:     {mape:.2f}%")
print(f"    90d End AUM:   ${future_90['yhat'].iloc[-1]/1e9:.2f}B")
print(f"\n  Random Forest NAV Anomaly Detection:")
print(f"    AUC-ROC:       {rf_auc:.4f}")
print(f"    Precision:     {rf_prec:.4f}")
print(f"    Recall:        {rf_rec:.4f}")
print(f"    F1 Score:      {rf_f1:.4f}")
print(f"\n  K-Means Fund Segmentation:")
print(f"    Clusters:      5 operational profiles")
print(f"    Funds:         {len(fund_features)}")
print(f"\n  Output files:")
print(f"    data/processed/aum_forecast_90d.csv")
print(f"    data/processed/nav_anomaly_scores.csv")
print(f"    data/processed/fund_segments.csv")
