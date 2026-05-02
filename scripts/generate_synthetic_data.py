"""
Generate synthetic financial datasets for pipeline testing.

Upgraded version — fixes four critical data quality issues:

  1. credit_risk.csv (was 200 rows, 31 defaults)
     → 2000 rows, 20-25% default rate (400-500 positives)
     → default_history properly CORRELATED with default_flag (not independent)
     → edge cases explicitly injected: high-income/high-DTI, low-credit/low-loan, etc.
     → feature space fully covered so XGBoost sees real decision boundaries

  2. credit_risk_features.csv
     → regenerated alongside raw (was identical stale copy)

  3. macroeconomic.csv (was 24 rows, Jan 2023–Dec 2024)
     → 60 rows, Jan 2020–Dec 2024
     → covers COVID shock (2020), recovery + inflation surge (2021),
       rate-hike cycle (2022), plateau (2023), and declining-rate period (2024)
     → meaningful macro variance instead of a single benign post-pandemic window

  4. market_volatility.csv (was 24 rows, Jan 2023–Dec 2024)
     → 60 rows, Jan 2020–Dec 2024
     → COVID VIX spike (March 2020: VIX ~70), 2022 bear market, 2023 recovery
     → realistic spread widening during stress periods
"""

import pandas as pd
import numpy as np
from pathlib import Path

rng = np.random.default_rng(42)   # seeded for reproducibility
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)


# ── 1. Credit Risk (2000 records) ─────────────────────────────────────────────

N = 2000

# --- Raw features --
credit_scores    = rng.integers(500, 820, N)
dti              = np.round(rng.uniform(0.10, 0.70, N), 2)
income           = rng.integers(25_000, 220_000, N)
loan_amount      = rng.integers(10_000, 600_000, N)
employment_years = np.round(rng.uniform(0.5, 30, N), 1)
loan_term        = rng.choice([60, 120, 180, 240, 360], N)
collateral_ratio = rng.uniform(0.75, 1.60, N)

# --- Edge-case injection (15% of records get extreme feature combinations) ---
# These are the profiles that a 200-row dataset almost certainly missed.
n_edge = int(N * 0.15)

# Segment 1: High income, high DTI (affluent overleveraged borrowers)
idx_hi_inc_hi_dti = rng.choice(N, n_edge // 3, replace=False)
income[idx_hi_inc_hi_dti]       = rng.integers(150_000, 220_000, n_edge // 3)
dti[idx_hi_inc_hi_dti]          = np.round(rng.uniform(0.55, 0.70, n_edge // 3), 2)

# Segment 2: Low credit score, small loan (thin-file borrowers)
idx_low_cs_low_loan = rng.choice(N, n_edge // 3, replace=False)
credit_scores[idx_low_cs_low_loan] = rng.integers(500, 560, n_edge // 3)
loan_amount[idx_low_cs_low_loan]   = rng.integers(10_000, 50_000, n_edge // 3)

# Segment 3: Long employment, very low DTI (ultra-safe borrowers)
idx_safe = rng.choice(N, n_edge // 3, replace=False)
employment_years[idx_safe] = np.round(rng.uniform(15, 30, n_edge // 3), 1)
dti[idx_safe]              = np.round(rng.uniform(0.10, 0.22, n_edge // 3), 2)
credit_scores[idx_safe]    = rng.integers(720, 820, n_edge // 3)

# --- default_history: now CORRELATED with risk profile (not random) ---
# Low-risk borrowers (cs > 700, dti < 0.30): 80% chance of history=0
# Mid-risk: roughly equal across 0/1/2
# High-risk (cs < 580 or dti > 0.55): strongly skewed toward 1/2
def assign_default_history(cs, dti_val, rng_inst):
    """Assign default_history to correlate with actual risk profile."""
    probs = []
    for c, d in zip(cs, dti_val):
        if c > 700 and d < 0.30:
            probs.append([0.82, 0.13, 0.05])     # mostly clean
        elif c < 580 or d > 0.55:
            probs.append([0.30, 0.40, 0.30])     # stressed — 1s and 2s likely
        else:
            probs.append([0.55, 0.30, 0.15])     # moderate
    result = []
    for p in probs:
        result.append(rng_inst.choice([0, 1, 2], p=p))
    return np.array(result)

default_history = assign_default_history(credit_scores, dti, rng)

# --- Default probability — multi-factor, produces 20-25% rate ---
# Each component is normalised to [0, 1]
cs_norm      = (850 - credit_scores) / 320        # lower CS = higher risk
dti_norm     = dti / 0.70                          # higher DTI = higher risk
dh_contrib   = default_history * 0.15             # 0=0, 1=0.15, 2=0.30 addition
emp_norm     = np.clip(1 - employment_years / 30, 0, 1)   # newer = riskier
income_norm  = np.clip(1 - income / 220_000, 0, 1)        # lower income = riskier

# Non-linear DTI: risk accelerates above 0.45
dti_nl = np.where(dti < 0.45, dti / 0.45, 1.0 + (dti - 0.45) / 0.25)
dti_nl = np.clip(dti_nl, 0, 2) / 2                # normalise to [0,1]

default_prob_raw = (
    0.28 * cs_norm
    + 0.25 * dti_nl
    + 0.20 * dh_contrib / 0.30      # scale to [0,1]
    + 0.15 * emp_norm
    + 0.12 * income_norm
)

# Add controlled noise and threshold -- produces ~22% default rate
noise        = rng.normal(0, 0.08, N)
default_flag = (default_prob_raw + noise > 0.65).astype(int)

# Force minimum defaults in each default_history cohort (ensures signal)
# Records with dh=2 that aren't defaulting: flip the bottom quartile
dh2_idx      = np.where(default_history == 2)[0]
dh2_no_def   = dh2_idx[default_flag[dh2_idx] == 0]
flip_count   = int(len(dh2_no_def) * 0.40)       # flip 40% of dh=2 non-defaults
flip_idx     = rng.choice(dh2_no_def, flip_count, replace=False)
default_flag[flip_idx] = 1

collateral_value = np.round(loan_amount * collateral_ratio, 2)

credit_df = pd.DataFrame({
    "loan_id":              [f"LOAN_{i:04d}" for i in range(N)],
    "borrower_income":      income,
    "debt_to_income_ratio": dti,
    "credit_score":         credit_scores,
    "loan_amount":          loan_amount,
    "loan_term_months":     loan_term,
    "employment_years":     employment_years,
    "default_history":      default_history,
    "collateral_value":     collateral_value,
    "default_flag":         default_flag
})

credit_df.to_csv("data/raw/credit_risk.csv", index=False)
credit_df.to_csv("data/processed/credit_risk_features.csv", index=False)

# Diagnostic output
dh_counts  = np.bincount(default_history, minlength=3)
def_rate   = default_flag.mean()
dh0_def    = default_flag[default_history == 0].mean()
dh1_def    = default_flag[default_history == 1].mean()
dh2_def    = default_flag[default_history == 2].mean()

print(f"credit_risk.csv: {len(credit_df)} records, {default_flag.sum()} defaults ({def_rate:.1%})")
print(f"  default_history counts: 0={dh_counts[0]}, 1={dh_counts[1]}, 2={dh_counts[2]}")
print(f"  default rate by history: dh=0 -> {dh0_def:.1%} | dh=1 -> {dh1_def:.1%} | dh=2 -> {dh2_def:.1%}")
print(f"  credit score range: {credit_scores.min()} - {credit_scores.max()}")
print(f"  DTI range: {dti.min():.2f} - {dti.max():.2f}")


# ── 2. Macroeconomic (60 months: Jan 2020 – Dec 2024) ────────────────────────
# Covers the full rate cycle: COVID shock → recovery → inflation surge →
# aggressive rate hikes → plateau → early easing
# Values are realistic to actual US macro conditions in each period.

dates_macro = pd.date_range("2020-01-01", periods=60, freq="ME")

def macro_series(dates, rng_inst):
    n      = len(dates)
    gdp    = np.zeros(n)
    inf    = np.zeros(n)
    rate   = np.zeros(n)
    unemp  = np.zeros(n)
    cci    = np.zeros(n)
    ipi    = np.zeros(n)

    for i, d in enumerate(dates):
        yr, mo = d.year, d.month

        # GDP growth rate (annualised, monthly)
        if yr == 2020 and mo in [3, 4]:
            gdp[i] = round(rng_inst.uniform(-0.090, -0.050), 4)   # COVID collapse
        elif yr == 2020 and mo in [5, 6, 7]:
            gdp[i] = round(rng_inst.uniform(0.040, 0.090), 4)     # sharp bounce
        elif yr == 2020:
            gdp[i] = round(rng_inst.uniform(-0.010, 0.025), 4)
        elif yr == 2021:
            gdp[i] = round(rng_inst.uniform(0.030, 0.065), 4)     # strong recovery
        elif yr == 2022:
            gdp[i] = round(rng_inst.uniform(0.005, 0.030), 4)     # slowing
        elif yr == 2023:
            gdp[i] = round(rng_inst.uniform(0.010, 0.035), 4)
        else:   # 2024
            gdp[i] = round(rng_inst.uniform(0.015, 0.035), 4)

        # Inflation rate
        if yr == 2020:
            inf[i] = round(rng_inst.uniform(0.005, 0.020), 4)     # deflationary
        elif yr == 2021 and mo < 7:
            inf[i] = round(rng_inst.uniform(0.020, 0.040), 4)
        elif yr == 2021:
            inf[i] = round(rng_inst.uniform(0.040, 0.065), 4)     # inflation building
        elif yr == 2022 and mo < 7:
            inf[i] = round(rng_inst.uniform(0.065, 0.095), 4)     # peak inflation
        elif yr == 2022:
            inf[i] = round(rng_inst.uniform(0.055, 0.080), 4)
        elif yr == 2023 and mo < 7:
            inf[i] = round(rng_inst.uniform(0.035, 0.060), 4)     # cooling
        elif yr == 2023:
            inf[i] = round(rng_inst.uniform(0.025, 0.045), 4)
        else:   # 2024
            inf[i] = round(rng_inst.uniform(0.022, 0.038), 4)     # near target

        # Fed funds / interest rate
        if yr == 2020 and mo <= 3:
            rate[i] = round(rng_inst.uniform(0.015, 0.025), 4)
        elif yr == 2020 and mo > 3:
            rate[i] = round(rng_inst.uniform(0.000, 0.005), 4)    # zero bound
        elif yr == 2021:
            rate[i] = round(rng_inst.uniform(0.000, 0.005), 4)    # still zero
        elif yr == 2022 and mo < 4:
            rate[i] = round(rng_inst.uniform(0.000, 0.010), 4)
        elif yr == 2022 and mo < 7:
            rate[i] = round(rng_inst.uniform(0.010, 0.030), 4)    # rapid hikes
        elif yr == 2022:
            rate[i] = round(rng_inst.uniform(0.030, 0.045), 4)
        elif yr == 2023 and mo < 7:
            rate[i] = round(rng_inst.uniform(0.045, 0.055), 4)    # plateau
        elif yr == 2023:
            rate[i] = round(rng_inst.uniform(0.053, 0.058), 4)
        else:   # 2024
            rate[i] = round(rng_inst.uniform(0.040, 0.058), 4)    # early cuts

        # Unemployment rate
        if yr == 2020 and mo in [3, 4, 5]:
            unemp[i] = round(rng_inst.uniform(0.060, 0.148), 4)   # pandemic spike
        elif yr == 2020:
            unemp[i] = round(rng_inst.uniform(0.065, 0.090), 4)
        elif yr == 2021:
            unemp[i] = round(rng_inst.uniform(0.040, 0.070), 4)
        elif yr == 2022:
            unemp[i] = round(rng_inst.uniform(0.035, 0.050), 4)
        elif yr == 2023:
            unemp[i] = round(rng_inst.uniform(0.036, 0.048), 4)
        else:
            unemp[i] = round(rng_inst.uniform(0.038, 0.052), 4)

        # Consumer Confidence Index (100 = neutral baseline)
        if yr == 2020 and mo in [3, 4]:
            cci[i] = round(rng_inst.uniform(62, 78), 2)
        elif yr == 2020:
            cci[i] = round(rng_inst.uniform(78, 95), 2)
        elif yr == 2021:
            cci[i] = round(rng_inst.uniform(92, 115), 2)
        elif yr == 2022:
            cci[i] = round(rng_inst.uniform(78, 100), 2)          # rate-hike anxiety
        elif yr == 2023:
            cci[i] = round(rng_inst.uniform(86, 108), 2)
        else:
            cci[i] = round(rng_inst.uniform(90, 112), 2)

        # Industrial Production Index
        if yr == 2020 and mo in [3, 4]:
            ipi[i] = round(rng_inst.uniform(78, 92), 2)
        elif yr == 2020:
            ipi[i] = round(rng_inst.uniform(88, 98), 2)
        elif yr == 2021:
            ipi[i] = round(rng_inst.uniform(98, 108), 2)
        elif yr == 2022:
            ipi[i] = round(rng_inst.uniform(100, 110), 2)
        else:
            ipi[i] = round(rng_inst.uniform(98, 108), 2)

    return gdp, inf, rate, unemp, cci, ipi

gdp, inf, rate, unemp, cci, ipi = macro_series(dates_macro, rng)

macro_df = pd.DataFrame({
    "date":                        dates_macro.strftime("%Y-%m-%d"),
    "gdp_growth_rate":             gdp,
    "inflation_rate":              inf,
    "interest_rate":               rate,
    "unemployment_rate":           unemp,
    "consumer_confidence_index":   cci,
    "industrial_production_index": ipi,
})
macro_df.to_csv("data/raw/macroeconomic.csv", index=False)
print(f"\nmacroeconomic.csv: {len(macro_df)} records (2020-01 to 2024-12)")
print(f"  interest_rate range: {rate.min():.3f} – {rate.max():.3f}")
print(f"  inflation range:     {inf.min():.3f} – {inf.max():.3f}")
print(f"  unemployment range:  {unemp.min():.3f} – {unemp.max():.3f}")


# ── 3. Market Volatility (60 months: Jan 2020 – Dec 2024) ────────────────────

def market_series(dates, rng_inst):
    n     = len(dates)
    vol   = np.zeros(n)    # volatility_index (0–100 scale)
    ret   = np.zeros(n)    # index monthly return
    vix   = np.zeros(n)    # VIX
    spd   = np.zeros(n)    # credit spread (bps)

    for i, d in enumerate(dates):
        yr, mo = d.year, d.month

        # VIX: key stress events
        if yr == 2020 and mo == 3:
            vix[i] = round(rng_inst.uniform(55, 78), 2)    # COVID spike
        elif yr == 2020 and mo in [2, 4]:
            vix[i] = round(rng_inst.uniform(35, 54), 2)
        elif yr == 2020 and mo in [5, 6]:
            vix[i] = round(rng_inst.uniform(24, 38), 2)
        elif yr == 2020:
            vix[i] = round(rng_inst.uniform(18, 30), 2)
        elif yr == 2021:
            vix[i] = round(rng_inst.uniform(15, 24), 2)    # calm recovery
        elif yr == 2022:
            vix[i] = round(rng_inst.uniform(20, 36), 2)    # rate-hike anxiety
        elif yr == 2023:
            vix[i] = round(rng_inst.uniform(14, 26), 2)
        else:
            vix[i] = round(rng_inst.uniform(12, 22), 2)

        # Volatility index (scaled to 12–35 normally, spiking in stress)
        if yr == 2020 and mo == 3:
            vol[i] = round(rng_inst.uniform(55, 80), 2)
        elif yr == 2020 and mo in [2, 4, 5]:
            vol[i] = round(rng_inst.uniform(30, 55), 2)
        elif yr == 2022:
            vol[i] = round(rng_inst.uniform(22, 40), 2)
        else:
            vol[i] = round(rng_inst.uniform(12, 30), 2)

        # Monthly index return
        if yr == 2020 and mo in [2, 3]:
            ret[i] = round(rng_inst.uniform(-0.16, -0.06), 4)   # COVID crash
        elif yr == 2020 and mo in [4, 5, 6]:
            ret[i] = round(rng_inst.uniform(0.04, 0.12), 4)     # recovery rally
        elif yr == 2020:
            ret[i] = round(rng_inst.uniform(-0.02, 0.06), 4)
        elif yr == 2021:
            ret[i] = round(rng_inst.uniform(0.01, 0.06), 4)     # bull run
        elif yr == 2022:
            ret[i] = round(rng_inst.uniform(-0.08, 0.02), 4)    # bear market
        elif yr == 2023:
            ret[i] = round(rng_inst.uniform(-0.03, 0.07), 4)
        else:
            ret[i] = round(rng_inst.uniform(-0.02, 0.06), 4)

        # Credit spread (wider = more stress)
        if yr == 2020 and mo in [3, 4]:
            spd[i] = round(rng_inst.uniform(4.0, 8.5), 3)
        elif yr == 2020:
            spd[i] = round(rng_inst.uniform(2.0, 4.5), 3)
        elif yr == 2021:
            spd[i] = round(rng_inst.uniform(0.5, 2.0), 3)
        elif yr == 2022:
            spd[i] = round(rng_inst.uniform(1.5, 3.5), 3)
        else:
            spd[i] = round(rng_inst.uniform(0.8, 2.5), 3)

    return vol, ret, vix, spd

vol, ret, vix, spd = market_series(dates_macro, rng)

market_df = pd.DataFrame({
    "date":             dates_macro.strftime("%Y-%m-%d"),
    "volatility_index": vol,
    "index_return":     ret,
    "vix":              vix,
    "spread":           spd,
})
market_df.to_csv("data/raw/market_volatility.csv", index=False)
print(f"\nmarket_volatility.csv: {len(market_df)} records (2020-01 to 2024-12)")
print(f"  VIX range:             {vix.min():.1f} – {vix.max():.1f}")
print(f"  volatility_index range:{vol.min():.1f} – {vol.max():.1f}")
print(f"  index_return range:    {ret.min():.4f} – {ret.max():.4f}")
print(f"  spread range:          {spd.min():.3f} – {spd.max():.3f}")

print("\n✓ All datasets regenerated in data/raw/ and data/processed/")
print("  Ready for pipeline ingestion and XGBoost training.")
