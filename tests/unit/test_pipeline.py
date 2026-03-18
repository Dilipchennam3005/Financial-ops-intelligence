"""
================================================================================
Pinnacle Asset Management — Financial Operations Intelligence Platform
Tests: Unit and Integration
================================================================================
Author: Dilip Chennam
================================================================================
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ── Fixtures
@pytest.fixture(scope='module')
def fund_master():
    return pd.read_csv('data/raw/fund_master.csv')

@pytest.fixture(scope='module')
def nav_daily():
    return pd.read_csv('data/raw/nav_daily.csv', parse_dates=['nav_date'])

@pytest.fixture(scope='module')
def cash_flows():
    return pd.read_csv('data/raw/cash_flows.csv', parse_dates=['transaction_date'])

@pytest.fixture(scope='module')
def monthly_perf():
    return pd.read_csv('data/raw/monthly_performance.csv')

@pytest.fixture(scope='module')
def fund_segments():
    return pd.read_csv('data/processed/fund_segments.csv')


# ============================================================
# DATA QUALITY TESTS
# ============================================================
class TestDataQuality:

    def test_fund_count(self, fund_master):
        assert len(fund_master) == 50

    def test_no_duplicate_fund_ids(self, fund_master):
        assert fund_master['fund_id'].nunique() == len(fund_master)

    def test_valid_asset_classes(self, fund_master):
        valid = {'Equity', 'Fixed Income', 'Multi-Asset', 'Alternative', 'Money Market'}
        assert set(fund_master['asset_class'].unique()).issubset(valid)

    def test_management_fee_range(self, fund_master):
        assert (fund_master['management_fee'] > 0).all()
        assert (fund_master['management_fee'] <= 3.0).all()

    def test_initial_aum_positive(self, fund_master):
        assert (fund_master['initial_aum'] > 0).all()

    def test_nav_record_count(self, nav_daily):
        assert len(nav_daily) == 65250

    def test_nav_positive(self, nav_daily):
        assert (nav_daily['nav'] > 0).all()

    def test_is_anomaly_binary(self, nav_daily):
        assert nav_daily['is_anomaly'].isin([0, 1]).all()

    def test_anomaly_rate_realistic(self, nav_daily):
        rate = nav_daily['is_anomaly'].mean()
        assert 0.005 <= rate <= 0.02

    def test_cash_flow_transaction_types(self, cash_flows):
        valid = {'Subscription', 'Redemption', 'Dividend',
                 'Management Fee', 'Performance Fee', 'Expense'}
        assert set(cash_flows['transaction_type'].unique()).issubset(valid)

    def test_transaction_status_valid(self, cash_flows):
        valid = {'Settled', 'Pending', 'Failed'}
        assert set(cash_flows['status'].unique()).issubset(valid)

    def test_monthly_perf_record_count(self, monthly_perf):
        assert len(monthly_perf) == 3000

    def test_sharpe_ratio_range(self, monthly_perf):
        assert monthly_perf['sharpe_ratio'].between(-5, 5).mean() > 0.95


# ============================================================
# FEATURE ENGINEERING TESTS
# ============================================================
class TestFeatureEngineering:

    def test_fund_segments_count(self, fund_segments):
        assert len(fund_segments) == 50

    def test_cluster_count(self, fund_segments):
        assert fund_segments['cluster_id'].nunique() == 5

    def test_cluster_labels_assigned(self, fund_segments):
        assert fund_segments['cluster_label'].notna().all()

    def test_avg_return_valid(self, fund_segments):
        assert fund_segments['avg_return'].between(-1, 1).all()

    def test_consistency_range(self, fund_segments):
        assert fund_segments['consistency'].between(0, 1).all()

    def test_high_sharpe_cluster_exists(self, fund_segments):
        max_sharpe_cluster = fund_segments.groupby('cluster_label')['avg_sharpe'].mean().idxmax()
        assert max_sharpe_cluster is not None

    def test_underperformer_cluster_negative_return(self, fund_segments):
        underperformers = fund_segments[fund_segments['cluster_label'] == 'Underperformers']
        if len(underperformers) > 0:
            assert underperformers['avg_return'].mean() < 0.01


# ============================================================
# PROPHET FORECAST TESTS
# ============================================================
class TestProphetForecast:

    def test_forecast_file_exists(self):
        assert os.path.exists('data/processed/aum_forecast_90d.csv')

    def test_forecast_record_count(self):
        df = pd.read_csv('data/processed/aum_forecast_90d.csv')
        assert len(df) == 90

    def test_forecast_columns(self):
        df = pd.read_csv('data/processed/aum_forecast_90d.csv')
        assert 'yhat' in df.columns
        assert 'yhat_lower' in df.columns
        assert 'yhat_upper' in df.columns

    def test_forecast_values_positive(self):
        df = pd.read_csv('data/processed/aum_forecast_90d.csv')
        assert (df['yhat'] > 0).all()

    def test_confidence_interval_valid(self):
        df = pd.read_csv('data/processed/aum_forecast_90d.csv')
        assert (df['yhat_upper'] >= df['yhat']).all()
        assert (df['yhat'] >= df['yhat_lower']).all()

    def test_forecast_trend_reasonable(self):
        df = pd.read_csv('data/processed/aum_forecast_90d.csv')
        start = df['yhat'].iloc[0]
        end   = df['yhat'].iloc[-1]
        change_pct = abs(end - start) / start
        assert change_pct < 0.50  # AUM shouldn't change by more than 50% in 90 days


# ============================================================
# ANOMALY DETECTION TESTS
# ============================================================
class TestAnomalyDetection:

    def test_anomaly_scores_file_exists(self):
        assert os.path.exists('data/processed/nav_anomaly_scores.csv')

    def test_anomaly_scores_count(self):
        df = pd.read_csv('data/processed/nav_anomaly_scores.csv')
        assert len(df) > 0

    def test_anomaly_prob_range(self):
        df = pd.read_csv('data/processed/nav_anomaly_scores.csv')
        assert (df['rf_anomaly_prob'].between(0, 1)).all()

    def test_flagged_anomaly_rate(self):
        df = pd.read_csv('data/processed/nav_anomaly_scores.csv')
        rate = df['rf_is_anomaly'].mean()
        assert 0.005 <= rate <= 0.05  # between 0.5% and 5%


# ============================================================
# PIPELINE INTEGRATION TESTS
# ============================================================
class TestPipelineIntegration:

    def test_all_funds_have_nav_records(self, fund_master, nav_daily):
        fund_ids_master = set(fund_master['fund_id'])
        fund_ids_nav    = set(nav_daily['fund_id'])
        assert fund_ids_master == fund_ids_nav

    def test_all_funds_have_monthly_perf(self, fund_master, monthly_perf):
        fund_ids_master  = set(fund_master['fund_id'])
        fund_ids_monthly = set(monthly_perf['fund_id'])
        assert fund_ids_master == fund_ids_monthly

    def test_all_funds_have_segments(self, fund_master, fund_segments):
        fund_ids_master   = set(fund_master['fund_id'])
        fund_ids_segments = set(fund_segments['fund_id'])
        assert fund_ids_master == fund_ids_segments

    def test_nav_date_range(self, nav_daily):
        assert nav_daily['nav_date'].min() >= pd.Timestamp('2020-01-01')
        assert nav_daily['nav_date'].max() <= pd.Timestamp('2024-12-31')

    def test_monthly_perf_covers_all_months(self, monthly_perf):
        months = monthly_perf['month'].nunique()
        assert months == 60  # 5 years x 12 months

    def test_cash_flow_amounts_nonzero(self, cash_flows):
        nonzero = cash_flows[cash_flows['amount'] != 0]
        assert len(nonzero) / len(cash_flows) > 0.95

    def test_subscriptions_positive(self, cash_flows):
        subs = cash_flows[cash_flows['transaction_type'] == 'Subscription']
        assert (subs['amount'] > 0).all()

    def test_redemptions_negative(self, cash_flows):
        reds = cash_flows[cash_flows['transaction_type'] == 'Redemption']
        assert (reds['amount'] < 0).all()
