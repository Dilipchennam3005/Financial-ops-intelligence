-- ============================================================
-- Pinnacle Asset Management — Financial Operations Intelligence
-- MySQL Schema DDL
-- File: mysql/01_schema.sql
-- Author: Dilip Chennam
-- ============================================================

CREATE DATABASE IF NOT EXISTS pinnacle_ops
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE pinnacle_ops;

-- ── Fund Master
CREATE TABLE IF NOT EXISTS fund_master (
    fund_id             VARCHAR(10)     PRIMARY KEY,
    fund_name           VARCHAR(100)    NOT NULL,
    asset_class         VARCHAR(30)     NOT NULL,
    region              VARCHAR(30),
    currency            VARCHAR(5),
    benchmark           VARCHAR(50),
    fund_manager        VARCHAR(50),
    inception_date      DATE,
    management_fee      DECIMAL(5,2),
    target_volatility   DECIMAL(8,4),
    initial_aum         DECIMAL(20,2),
    status              VARCHAR(10),
    created_at          TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

-- ── Daily NAV
CREATE TABLE IF NOT EXISTS nav_daily (
    id                  BIGINT          AUTO_INCREMENT PRIMARY KEY,
    fund_id             VARCHAR(10)     NOT NULL,
    nav_date            DATE            NOT NULL,
    nav                 DECIMAL(12,4),
    aum                 DECIMAL(20,2),
    daily_return        DECIMAL(10,6),
    benchmark_return    DECIMAL(10,6),
    active_return       DECIMAL(10,6),
    is_anomaly          TINYINT(1)      DEFAULT 0,
    FOREIGN KEY (fund_id) REFERENCES fund_master(fund_id),
    INDEX idx_fund_date (fund_id, nav_date),
    INDEX idx_date      (nav_date),
    UNIQUE KEY uk_fund_date (fund_id, nav_date)
);

-- ── Cash Flows
CREATE TABLE IF NOT EXISTS cash_flows (
    transaction_id      VARCHAR(15)     PRIMARY KEY,
    fund_id             VARCHAR(10)     NOT NULL,
    transaction_date    DATE            NOT NULL,
    transaction_type    VARCHAR(20),
    amount              DECIMAL(20,2),
    currency            VARCHAR(5),
    counterparty        VARCHAR(10),
    status              VARCHAR(10),
    FOREIGN KEY (fund_id) REFERENCES fund_master(fund_id),
    INDEX idx_fund_date (fund_id, transaction_date),
    INDEX idx_type      (transaction_type)
);

-- ── Trades
CREATE TABLE IF NOT EXISTS trades (
    trade_id            VARCHAR(15)     PRIMARY KEY,
    fund_id             VARCHAR(10)     NOT NULL,
    trade_date          DATE            NOT NULL,
    security            VARCHAR(20),
    security_type       VARCHAR(20),
    direction           VARCHAR(5),
    quantity            INT,
    price               DECIMAL(12,4),
    notional            DECIMAL(20,2),
    commission          DECIMAL(10,2),
    status              VARCHAR(10),
    FOREIGN KEY (fund_id) REFERENCES fund_master(fund_id),
    INDEX idx_fund_date (fund_id, trade_date),
    INDEX idx_security  (security)
);

-- ── Monthly Performance
CREATE TABLE IF NOT EXISTS monthly_performance (
    id                  BIGINT          AUTO_INCREMENT PRIMARY KEY,
    fund_id             VARCHAR(10)     NOT NULL,
    asset_class         VARCHAR(30),
    month               VARCHAR(10),
    monthly_return      DECIMAL(10,6),
    benchmark_return    DECIMAL(10,6),
    active_return       DECIMAL(10,6),
    avg_aum             DECIMAL(20,2),
    month_end_nav       DECIMAL(12,4),
    anomaly_count       INT             DEFAULT 0,
    sharpe_ratio        DECIMAL(10,4),
    FOREIGN KEY (fund_id) REFERENCES fund_master(fund_id),
    INDEX idx_fund_month (fund_id, month),
    INDEX idx_month      (month)
);

-- ── ML Model Scores
CREATE TABLE IF NOT EXISTS ml_scores (
    score_id            BIGINT          AUTO_INCREMENT PRIMARY KEY,
    fund_id             VARCHAR(10)     NOT NULL,
    score_date          DATE            NOT NULL,
    model_type          VARCHAR(30),
    model_version       VARCHAR(10),
    anomaly_score       DECIMAL(10,6),
    is_anomaly          TINYINT(1),
    cluster_id          INT,
    cluster_label       VARCHAR(30),
    forecast_30d        DECIMAL(20,2),
    forecast_60d        DECIMAL(20,2),
    forecast_90d        DECIMAL(20,2),
    FOREIGN KEY (fund_id) REFERENCES fund_master(fund_id),
    INDEX idx_fund_date (fund_id, score_date)
);

-- ============================================================
-- ANALYTICAL VIEWS
-- ============================================================

CREATE OR REPLACE VIEW vw_aum_by_asset_class AS
SELECT
    f.asset_class,
    COUNT(DISTINCT f.fund_id)               AS fund_count,
    ROUND(SUM(n.aum)/1e9, 2)               AS total_aum_billions,
    ROUND(AVG(n.aum)/1e6, 2)               AS avg_fund_aum_millions,
    ROUND(AVG(n.daily_return)*252*100, 2)  AS annualized_return_pct,
    ROUND(AVG(n.active_return)*252*100, 2) AS avg_active_return_pct,
    SUM(n.is_anomaly)                       AS total_anomalies
FROM fund_master f
JOIN nav_daily n ON f.fund_id = n.fund_id
WHERE n.nav_date = (SELECT MAX(nav_date) FROM nav_daily)
GROUP BY f.asset_class;

CREATE OR REPLACE VIEW vw_fund_performance AS
SELECT
    f.fund_id,
    f.fund_name,
    f.asset_class,
    f.benchmark,
    f.fund_manager,
    ROUND(SUM(m.monthly_return)*100, 2)     AS total_return_pct,
    ROUND(SUM(m.benchmark_return)*100, 2)   AS benchmark_return_pct,
    ROUND(SUM(m.active_return)*100, 2)      AS total_active_return_pct,
    ROUND(AVG(m.sharpe_ratio), 4)           AS avg_sharpe_ratio,
    ROUND(AVG(m.avg_aum)/1e6, 2)           AS avg_aum_millions,
    SUM(m.anomaly_count)                    AS total_anomalies
FROM fund_master f
JOIN monthly_performance m ON f.fund_id = m.fund_id
GROUP BY f.fund_id, f.fund_name, f.asset_class, f.benchmark, f.fund_manager;

CREATE OR REPLACE VIEW vw_cashflow_summary AS
SELECT
    f.fund_id,
    f.fund_name,
    f.asset_class,
    c.transaction_type,
    COUNT(*)                                AS transaction_count,
    ROUND(SUM(c.amount)/1e6, 2)            AS total_amount_millions,
    ROUND(AVG(c.amount)/1e3, 2)            AS avg_amount_thousands,
    SUM(CASE WHEN c.status='Failed' THEN 1 ELSE 0 END) AS failed_count
FROM fund_master f
JOIN cash_flows c ON f.fund_id = c.fund_id
GROUP BY f.fund_id, f.fund_name, f.asset_class, c.transaction_type;

CREATE OR REPLACE VIEW vw_nav_anomalies AS
SELECT
    f.fund_id,
    f.fund_name,
    f.asset_class,
    f.fund_manager,
    COUNT(*)                                AS total_nav_records,
    SUM(n.is_anomaly)                       AS anomaly_count,
    ROUND(SUM(n.is_anomaly)/COUNT(*)*100,2) AS anomaly_rate_pct,
    ROUND(AVG(n.aum)/1e6, 2)               AS avg_aum_millions
FROM fund_master f
JOIN nav_daily n ON f.fund_id = n.fund_id
GROUP BY f.fund_id, f.fund_name, f.asset_class, f.fund_manager
ORDER BY anomaly_count DESC;
