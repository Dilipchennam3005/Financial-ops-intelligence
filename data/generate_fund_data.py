"""
================================================================================
Pinnacle Asset Management — Financial Operations Intelligence Platform
Dataset Generator
================================================================================
Generates realistic fund operations data including:
  - Fund master data (50 funds across 5 asset classes)
  - Daily NAV records (2020–2024)
  - Cash flow transactions
  - Trade records
  - Benchmark performance

Author: Dilip Chennam
================================================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
OUTPUT_DIR = 'data/raw'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Date range
START_DATE = datetime(2020, 1, 1)
END_DATE   = datetime(2024, 12, 31)
DATES      = pd.bdate_range(START_DATE, END_DATE)  # business days only

print("=" * 60)
print("Pinnacle Asset Management — Data Generator")
print("=" * 60)

# ============================================================
# 1. FUND MASTER DATA
# ============================================================
print("\n[1/5] Generating fund master data...")

ASSET_CLASSES = {
    'Equity':          {'count': 15, 'base_aum': (200e6,  2e9),  'fee': (0.50, 1.20), 'vol': (0.12, 0.25)},
    'Fixed Income':    {'count': 12, 'base_aum': (150e6,  1.5e9),'fee': (0.25, 0.65), 'vol': (0.04, 0.10)},
    'Multi-Asset':     {'count': 10, 'base_aum': (100e6,  1e9),  'fee': (0.60, 1.00), 'vol': (0.08, 0.16)},
    'Alternative':     {'count':  8, 'base_aum': (50e6,   500e6),'fee': (1.00, 2.00), 'vol': (0.15, 0.30)},
    'Money Market':    {'count':  5, 'base_aum': (500e6,  5e9),  'fee': (0.10, 0.25), 'vol': (0.01, 0.03)},
}

REGIONS       = ['US', 'Global', 'Europe', 'Asia-Pacific', 'Emerging Markets']
FUND_MANAGERS = ['Sarah Chen', 'Michael Torres', 'Priya Patel', 'James Whitfield', 'Lisa Nakamura']
CURRENCIES    = ['USD', 'USD', 'USD', 'EUR', 'GBP']
BENCHMARKS    = {
    'Equity':       ['S&P 500', 'Russell 1000', 'MSCI World', 'MSCI EAFE', 'MSCI EM'],
    'Fixed Income': ['Bloomberg US Agg', 'Bloomberg Global Agg', 'ICE BofA Corp', 'Bloomberg EM'],
    'Multi-Asset':  ['60/40 Blend', 'MSCI ACWI', 'CPI+3%'],
    'Alternative':  ['HFRI Fund Weighted', 'HFRI Macro', 'Credit Suisse HF'],
    'Money Market': ['Fed Funds Rate', '3M T-Bill', 'SOFR'],
}

funds = []
fund_id = 1
for asset_class, config in ASSET_CLASSES.items():
    for i in range(config['count']):
        aum         = np.random.uniform(*config['base_aum'])
        inception   = START_DATE - timedelta(days=np.random.randint(365, 3650))
        funds.append({
            'fund_id':          f'PAM{str(fund_id).zfill(4)}',
            'fund_name':        f'Pinnacle {asset_class} Fund {i+1:02d}',
            'asset_class':      asset_class,
            'region':           np.random.choice(REGIONS),
            'currency':         np.random.choice(CURRENCIES),
            'benchmark':        np.random.choice(BENCHMARKS[asset_class]),
            'fund_manager':     np.random.choice(FUND_MANAGERS),
            'inception_date':   inception.strftime('%Y-%m-%d'),
            'management_fee':   round(np.random.uniform(*config['fee']), 2),
            'target_volatility':round(np.random.uniform(*config['vol']), 4),
            'initial_aum':      round(aum, 2),
            'status':           np.random.choice(['Active', 'Active', 'Active', 'Closed'], p=[0.85, 0.05, 0.05, 0.05]),
        })
        fund_id += 1

funds_df = pd.DataFrame(funds)
funds_df.to_csv(f'{OUTPUT_DIR}/fund_master.csv', index=False)
print(f"  ✅ {len(funds_df)} funds across {len(ASSET_CLASSES)} asset classes")
print(f"     Total initial AUM: ${funds_df['initial_aum'].sum()/1e9:.1f}B")

# ============================================================
# 2. DAILY NAV RECORDS
# ============================================================
print("\n[2/5] Generating daily NAV records...")

nav_records = []
vol_config  = {ac: cfg['vol'] for ac, cfg in ASSET_CLASSES.items()}

# Market regime factors
def get_market_factor(date, asset_class):
    year, month = date.year, date.month
    # COVID crash March 2020
    if year == 2020 and month in [3, 4]:
        return -0.0025 if asset_class != 'Money Market' else 0.0001
    # Recovery 2020-2021
    if year in [2020, 2021] and month > 4:
        return 0.0008
    # Rate hike pressure 2022
    if year == 2022:
        return -0.0004 if asset_class in ['Equity', 'Fixed Income'] else 0.0002
    # Recovery 2023-2024
    if year in [2023, 2024]:
        return 0.0005
    return 0.0003

for _, fund in funds_df.iterrows():
    nav = fund['initial_aum'] / np.random.randint(1000000, 50000000)  # shares outstanding
    nav = round(nav, 4)
    aum = fund['initial_aum']
    vol = np.mean(vol_config[fund['asset_class']]) / np.sqrt(252)

    for date in DATES:
        mkt_factor   = get_market_factor(date, fund['asset_class'])
        daily_return = np.random.normal(mkt_factor, vol)
        nav          = round(nav * (1 + daily_return), 4)
        aum          = round(aum * (1 + daily_return) + np.random.normal(0, aum * 0.001), 2)
        aum          = max(aum, 1e6)

        # Benchmark return (correlated with fund)
        bench_return = daily_return * np.random.uniform(0.7, 1.3) + np.random.normal(0, vol * 0.2)
        active_return = daily_return - bench_return

        # NAV anomaly injection (1% of records)
        is_anomaly = 0
        if np.random.random() < 0.01:
            nav        *= np.random.choice([0.95, 1.05])
            is_anomaly  = 1

        nav_records.append({
            'fund_id':          fund['fund_id'],
            'nav_date':         date.strftime('%Y-%m-%d'),
            'nav':              nav,
            'aum':              round(aum, 2),
            'daily_return':     round(daily_return, 6),
            'benchmark_return': round(bench_return, 6),
            'active_return':    round(active_return, 6),
            'is_anomaly':       is_anomaly,
        })

nav_df = pd.DataFrame(nav_records)
nav_df.to_csv(f'{OUTPUT_DIR}/nav_daily.csv', index=False)
print(f"  ✅ {len(nav_df):,} NAV records | {nav_df['is_anomaly'].sum():,} injected anomalies")

# ============================================================
# 3. CASH FLOW TRANSACTIONS
# ============================================================
print("\n[3/5] Generating cash flow transactions...")

cf_records = []
cf_types   = ['Subscription', 'Redemption', 'Dividend', 'Management Fee', 'Performance Fee', 'Expense']
cf_weights = [0.30, 0.28, 0.15, 0.12, 0.08, 0.07]

for _, fund in funds_df.iterrows():
    # 3–8 transactions per month per fund
    for date in pd.date_range(START_DATE, END_DATE, freq='ME'):
        n_transactions = np.random.randint(3, 9)
        for _ in range(n_transactions):
            cf_type = np.random.choice(cf_types, p=cf_weights)
            if cf_type == 'Subscription':
                amount = np.random.uniform(100000, 5000000)
            elif cf_type == 'Redemption':
                amount = -np.random.uniform(100000, 3000000)
            elif cf_type == 'Dividend':
                amount = -np.random.uniform(50000, 500000)
            elif cf_type == 'Management Fee':
                amount = -fund['management_fee'] / 100 / 12 * fund['initial_aum'] * np.random.uniform(0.9, 1.1)
            elif cf_type == 'Performance Fee':
                amount = -np.random.uniform(10000, 200000) if np.random.random() > 0.6 else 0
            else:
                amount = -np.random.uniform(5000, 50000)

            cf_records.append({
                'transaction_id':   f'TXN{len(cf_records)+1:08d}',
                'fund_id':          fund['fund_id'],
                'transaction_date': (date - timedelta(days=np.random.randint(0, 28))).strftime('%Y-%m-%d'),
                'transaction_type': cf_type,
                'amount':           round(amount, 2),
                'currency':         fund['currency'],
                'counterparty':     f'CP{np.random.randint(1, 50):03d}',
                'status':           np.random.choice(['Settled', 'Settled', 'Settled', 'Pending', 'Failed'], p=[0.88, 0.05, 0.04, 0.02, 0.01]),
            })

cf_df = pd.DataFrame(cf_records)
cf_df.to_csv(f'{OUTPUT_DIR}/cash_flows.csv', index=False)
print(f"  ✅ {len(cf_df):,} cash flow transactions")

# ============================================================
# 4. TRADE RECORDS
# ============================================================
print("\n[4/5] Generating trade records...")

SECURITIES = [
    ('AAPL', 'Equity'), ('MSFT', 'Equity'), ('JPM', 'Equity'), ('GS', 'Equity'),
    ('BRK.B', 'Equity'), ('GOOGL', 'Equity'), ('AMZN', 'Equity'), ('META', 'Equity'),
    ('US10Y', 'Fixed Income'), ('US2Y', 'Fixed Income'), ('IG_CORP', 'Fixed Income'),
    ('HY_CORP', 'Fixed Income'), ('TIPS', 'Fixed Income'),
    ('GOLD', 'Alternative'), ('OIL', 'Alternative'), ('BTC', 'Alternative'),
]

trade_records = []
for _, fund in funds_df.iterrows():
    n_trades = np.random.randint(200, 800)
    for _ in range(n_trades):
        security, sec_type = SECURITIES[np.random.randint(0, len(SECURITIES))]
        trade_date = START_DATE + timedelta(days=np.random.randint(0, (END_DATE - START_DATE).days))
        price      = np.random.uniform(10, 500)
        quantity   = np.random.randint(100, 10000)
        direction  = np.random.choice(['Buy', 'Sell'])
        notional   = round(price * quantity, 2)

        trade_records.append({
            'trade_id':     f'TRD{len(trade_records)+1:08d}',
            'fund_id':      fund['fund_id'],
            'trade_date':   trade_date.strftime('%Y-%m-%d'),
            'security':     security,
            'security_type':sec_type,
            'direction':    direction,
            'quantity':     quantity,
            'price':        round(price, 4),
            'notional':     notional,
            'commission':   round(notional * np.random.uniform(0.0001, 0.0010), 2),
            'status':       np.random.choice(['Settled', 'Settled', 'Pending', 'Failed'], p=[0.90, 0.06, 0.03, 0.01]),
        })

trades_df = pd.DataFrame(trade_records)
trades_df.to_csv(f'{OUTPUT_DIR}/trades.csv', index=False)
print(f"  ✅ {len(trades_df):,} trade records")

# ============================================================
# 5. MONTHLY PERFORMANCE SUMMARY
# ============================================================
print("\n[5/5] Generating monthly performance summary...")

monthly_records = []
for _, fund in funds_df.iterrows():
    fund_nav = nav_df[nav_df['fund_id'] == fund['fund_id']].copy()
    fund_nav['nav_date'] = pd.to_datetime(fund_nav['nav_date'])
    fund_nav['month']    = fund_nav['nav_date'].dt.to_period('M')

    for month, group in fund_nav.groupby('month'):
        monthly_return    = group['daily_return'].sum()
        monthly_bench     = group['benchmark_return'].sum()
        monthly_active    = monthly_return - monthly_bench
        avg_aum           = group['aum'].mean()
        month_end_nav     = group['nav'].iloc[-1]
        anomaly_count     = group['is_anomaly'].sum()

        monthly_records.append({
            'fund_id':          fund['fund_id'],
            'asset_class':      fund['asset_class'],
            'month':            str(month),
            'monthly_return':   round(monthly_return, 6),
            'benchmark_return': round(monthly_bench, 6),
            'active_return':    round(monthly_active, 6),
            'avg_aum':          round(avg_aum, 2),
            'month_end_nav':    round(month_end_nav, 4),
            'anomaly_count':    int(anomaly_count),
            'sharpe_ratio':     round(monthly_return / (group['daily_return'].std() * np.sqrt(252) + 1e-8), 4),
        })

monthly_df = pd.DataFrame(monthly_records)
monthly_df.to_csv(f'{OUTPUT_DIR}/monthly_performance.csv', index=False)
print(f"  ✅ {len(monthly_df):,} monthly performance records")

# ── Summary
print("\n" + "=" * 60)
print("GENERATION COMPLETE")
print("=" * 60)
print(f"  Funds:               {len(funds_df)}")
print(f"  NAV records:         {len(nav_df):,}")
print(f"  Cash flow txns:      {len(cf_df):,}")
print(f"  Trade records:       {len(trades_df):,}")
print(f"  Monthly summaries:   {len(monthly_df):,}")
print(f"  Total AUM (initial): ${funds_df['initial_aum'].sum()/1e9:.1f}B")
print(f"  NAV anomalies:       {nav_df['is_anomaly'].sum():,}")
print(f"  Date range:          2020–2024")
print(f"\n  Output: {OUTPUT_DIR}/")
