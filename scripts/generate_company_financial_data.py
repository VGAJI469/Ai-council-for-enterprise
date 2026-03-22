"""
Generates synthetic company financial health dataset
for training the risk model on corporate financial standing.
"""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
N = 5000

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Company identifiers
company_ids = [f"COMP_{i:05d}" for i in range(N)]
sectors = np.random.choice([
    'Technology', 'Healthcare', 'Finance', 'Retail',
    'Manufacturing', 'Energy', 'Real Estate', 'Telecom'
], N)

# Revenue and profitability
annual_revenue          = np.random.uniform(1_000_000, 500_000_000, N)
revenue_growth_rate     = np.random.uniform(-0.30, 0.60, N)
gross_profit_margin     = np.random.uniform(0.05, 0.75, N)
net_profit_margin       = np.random.uniform(-0.20, 0.40, N)
ebitda_margin           = np.random.uniform(-0.10, 0.50, N)
operating_profit_margin = np.random.uniform(-0.15, 0.45, N)

# Debt and liabilities
total_debt              = annual_revenue * np.random.uniform(0.1, 3.5, N)
debt_to_equity_ratio    = np.random.uniform(0.1, 5.0, N)
debt_to_revenue_ratio   = total_debt / annual_revenue
interest_coverage_ratio = np.random.uniform(0.5, 20.0, N)
long_term_debt_ratio    = np.random.uniform(0.1, 0.9, N)

# Cash flow and liquidity
operating_cash_flow     = annual_revenue * np.random.uniform(-0.10, 0.30, N)
free_cash_flow          = operating_cash_flow * np.random.uniform(0.5, 1.2, N)
current_ratio           = np.random.uniform(0.5, 4.0, N)
quick_ratio             = current_ratio * np.random.uniform(0.5, 0.95, N)
cash_reserve_months     = np.random.uniform(0.5, 24.0, N)
cash_flow_to_debt       = operating_cash_flow / (total_debt + 1)

# Market and growth
market_share_percent    = np.random.uniform(0.1, 35.0, N)
market_growth_rate      = np.random.uniform(-0.10, 0.40, N)
customer_retention_rate = np.random.uniform(0.50, 0.98, N)
revenue_concentration   = np.random.uniform(0.10, 0.90, N)

# Operational efficiency
employee_count          = np.random.randint(10, 50000, N)
revenue_per_employee    = annual_revenue / employee_count
asset_turnover_ratio    = np.random.uniform(0.2, 3.0, N)
inventory_turnover      = np.random.uniform(2.0, 20.0, N)
return_on_assets        = np.random.uniform(-0.15, 0.30, N)
return_on_equity        = np.random.uniform(-0.30, 0.50, N)

# Risk indicators
years_in_operation      = np.random.randint(1, 80, N)
credit_rating_score     = np.random.uniform(300, 850, N)
bankruptcy_history      = np.random.choice([0, 0, 0, 0, 1], N)
regulatory_violations   = np.random.choice([0, 0, 0, 1, 2], N)
management_score        = np.random.uniform(1.0, 10.0, N)
esg_score               = np.random.uniform(10.0, 90.0, N)

# Target variable — financial distress risk
# Higher risk if: low margins, high debt, low cash, negative growth
distress_score = (
    (debt_to_equity_ratio / 5.0) * 0.20 +
    np.clip(-net_profit_margin, 0, 0.2) / 0.2 * 0.20 +
    (1 - np.clip(current_ratio / 4.0, 0, 1)) * 0.15 +
    np.clip(-revenue_growth_rate, 0, 0.3) / 0.3 * 0.15 +
    (1 - np.clip(interest_coverage_ratio / 20.0, 0, 1)) * 0.10 +
    np.clip(-free_cash_flow / (annual_revenue + 1), 0, 0.1) / 0.1 * 0.10 +
    bankruptcy_history * 0.05 +
    (regulatory_violations / 2.0) * 0.05
)

noise             = np.random.normal(0, 0.05, N)
financial_distress= (np.clip(distress_score + noise, 0, 1) > 0.45).astype(int)

df = pd.DataFrame({
    'company_id':              company_ids,
    'sector':                  sectors,
    'annual_revenue':          np.round(annual_revenue, 2),
    'revenue_growth_rate':     np.round(revenue_growth_rate, 4),
    'gross_profit_margin':     np.round(gross_profit_margin, 4),
    'net_profit_margin':       np.round(net_profit_margin, 4),
    'ebitda_margin':           np.round(ebitda_margin, 4),
    'operating_profit_margin': np.round(operating_profit_margin, 4),
    'total_debt':              np.round(total_debt, 2),
    'debt_to_equity_ratio':    np.round(debt_to_equity_ratio, 4),
    'debt_to_revenue_ratio':   np.round(debt_to_revenue_ratio, 4),
    'interest_coverage_ratio': np.round(interest_coverage_ratio, 4),
    'long_term_debt_ratio':    np.round(long_term_debt_ratio, 4),
    'operating_cash_flow':     np.round(operating_cash_flow, 2),
    'free_cash_flow':          np.round(free_cash_flow, 2),
    'current_ratio':           np.round(current_ratio, 4),
    'quick_ratio':             np.round(quick_ratio, 4),
    'cash_reserve_months':     np.round(cash_reserve_months, 2),
    'cash_flow_to_debt':       np.round(cash_flow_to_debt, 4),
    'market_share_percent':    np.round(market_share_percent, 4),
    'market_growth_rate':      np.round(market_growth_rate, 4),
    'customer_retention_rate': np.round(customer_retention_rate, 4),
    'revenue_concentration':   np.round(revenue_concentration, 4),
    'employee_count':          employee_count,
    'revenue_per_employee':    np.round(revenue_per_employee, 2),
    'asset_turnover_ratio':    np.round(asset_turnover_ratio, 4),
    'inventory_turnover':      np.round(inventory_turnover, 4),
    'return_on_assets':        np.round(return_on_assets, 4),
    'return_on_equity':        np.round(return_on_equity, 4),
    'years_in_operation':      years_in_operation,
    'credit_rating_score':     np.round(credit_rating_score, 2),
    'bankruptcy_history':      bankruptcy_history,
    'regulatory_violations':   regulatory_violations,
    'management_score':        np.round(management_score, 2),
    'esg_score':               np.round(esg_score, 2),
    'financial_distress':      financial_distress,
})

df.to_csv('data/raw/company_financial_health.csv', index=False)
df.to_csv('data/processed/company_financial_features.csv', index=False)

print(f'Company records generated  : {len(df)}')
print(f'Financial distress cases   : {df["financial_distress"].sum()}')
print(f'Healthy companies          : {(df["financial_distress"]==0).sum()}')
print(f'Distress rate              : {df["financial_distress"].mean():.2%}')
print(f'Features                   : {len(df.columns) - 3}')
print(f'Sectors covered            : {df["sector"].nunique()}')
print()
print('Saved to data/raw/company_financial_health.csv')
print('Saved to data/processed/company_financial_features.csv')
