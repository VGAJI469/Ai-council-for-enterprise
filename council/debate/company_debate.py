"""
Company Financial Risk Debate
Runs the full boardroom debate on a company financial profile.
Agents argue about the actual financial standing of the company.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import joblib
import numpy as np
from agents.evolution.agent_factory import AgentFactory
from council.voting.weighted_aggregator import WeightedAggregator
from council.debate.boardroom_debate import run_debate
from pipeline.aggregation.feature_builder import FeatureBuilder

ROLE_TITLES = {
    'strategic_growth':      'CEO',
    'financial_stability':   'CFO',
    'market_expansion':      'Chief Marketing Officer',
    'reputation_risk':       'PR Director',
    'regulatory_compliance': 'Legal Counsel',
}

def assess_company(company_profile: dict):
    builder  = FeatureBuilder()
    features = builder.build_company_profile(company_profile)

    try:
        art    = joblib.load('models/company_risk_model.pkl')
        model  = art['model']
        scaler = art['scaler']
        cols   = art['features']
        row    = np.array([[company_profile.get(c, 0) for c in cols]])
        scaled = scaler.transform(row)
        ml_risk= model.predict_proba(scaled)[0][1]
        features['ml_risk_score'] = round(ml_risk, 4)
        print(f'  ML Model Risk Score: {ml_risk:.4f}')
    except Exception as e:
        print(f'  ML model not loaded: {e}')
        features['ml_risk_score'] = features.get('default_probability', 0.5)

    name    = company_profile.get('company_name', 'the company')
    sector  = company_profile.get('sector', 'Unknown')
    revenue = company_profile.get('annual_revenue', 0)
    npm     = company_profile.get('net_profit_margin', 0)
    dte     = company_profile.get('debt_to_equity_ratio', 0)
    fcf     = company_profile.get('free_cash_flow', 0)
    growth  = company_profile.get('revenue_growth_rate', 0)
    cr      = company_profile.get('current_ratio', 0)
    ms      = company_profile.get('market_share_percent', 0)
    esg     = company_profile.get('esg_score', 50)
    ml_risk = features['ml_risk_score']

    motion = (
        f'Based on the financial standing of {name} ({sector} sector), '
        f'with annual revenue of ${revenue:,.0f}, '
        f'net profit margin of {npm:.1%}, '
        f'debt to equity ratio of {dte:.2f}, '
        f'free cash flow of ${fcf:,.0f}, '
        f'revenue growth of {growth:.1%}, '
        f'current ratio of {cr:.2f}, '
        f'market share of {ms:.1f}%, '
        f'ESG score of {esg:.0f}, '
        f'and an ML-predicted financial distress risk of {ml_risk:.2%} -- '
        f'should the board classify this company as HIGH RISK, MEDIUM RISK, or LOW RISK '
        f'and what immediate actions should be taken?'
    )

    return run_debate(motion, features)


if __name__ == '__main__':

    company = {
        'company_name':            'TechNova Solutions',
        'sector':                  'Technology',
        'annual_revenue':          85_000_000,
        'revenue_growth_rate':     0.12,
        'gross_profit_margin':     0.58,
        'net_profit_margin':       0.08,
        'ebitda_margin':           0.14,
        'operating_profit_margin': 0.11,
        'total_debt':              42_000_000,
        'debt_to_equity_ratio':    1.8,
        'debt_to_revenue_ratio':   0.49,
        'interest_coverage_ratio': 4.2,
        'long_term_debt_ratio':    0.55,
        'operating_cash_flow':     9_500_000,
        'free_cash_flow':          6_200_000,
        'current_ratio':           1.6,
        'quick_ratio':             1.2,
        'cash_reserve_months':     4.5,
        'cash_flow_to_debt':       0.22,
        'market_share_percent':    8.4,
        'market_growth_rate':      0.22,
        'customer_retention_rate': 0.84,
        'revenue_concentration':   0.38,
        'employee_count':          420,
        'revenue_per_employee':    202_381,
        'asset_turnover_ratio':    0.95,
        'inventory_turnover':      8.2,
        'return_on_assets':        0.09,
        'return_on_equity':        0.16,
        'years_in_operation':      12,
        'credit_rating_score':     680,
        'bankruptcy_history':      0,
        'regulatory_violations':   1,
        'management_score':        7.2,
        'esg_score':               64.0,
    }

    assess_company(company)
