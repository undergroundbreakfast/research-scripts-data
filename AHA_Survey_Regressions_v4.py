#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
"""
AI & Robotics Adoption vs. County-Level Outcomes
Author:  Aaron Johnson
Date  :  2025-07-01

THIS CODE WORKS BUT HAS PLOT FORMATTING ISSUES - UPDATED FOR ADJPD

Purpose:
• Pulls data from vw_county_tech_summary + vw_conceptual_model
• Runs 15 bivariate OLS regressions
• Saves:
    - logs/ai_robotics_regressions.log          (run-time log)
    - output/regression_summary.csv             (tabular results)
    - figs/{IV}__{DV}.png                       (scatter + line)

NB: All DB columns are stored as text → cast to NUMERIC in SQL for accuracy.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sqlalchemy import create_engine, text
from scipy.stats import pearsonr

# --------------------------------------------------------------------------------------
# Configuration — edit only this section if you need new pairs or labels
# --------------------------------------------------------------------------------------
IV_DV_PAIRS = [
    # (iv_col, dv_col, iv_label, dv_label)
    ("pct_wfaipsn_enabled", "crude_rate",
     "AI – Predict Staff Needs (%)", "All-Cause Mortality Crude Rate"),
    ("pct_wfaippd_enabled", "crude_rate",
     "AI – Predict Patient Demand (%)", "All-Cause Mortality Crude Rate"),
    ("pct_wfaiss_enabled", "crude_rate",
     "AI – Staff Scheduling (%)", "All-Cause Mortality Crude Rate"),
    ("pct_wfaiart_enabled", "crude_rate",
     "AI – Automate Routine Tasks (%)", "All-Cause Mortality Crude Rate"),
    ("pct_wfaioacw_enabled", "crude_rate",
     "AI – Optimize Admin./Clinical WF (%)", "All-Cause Mortality Crude Rate"),
    ("pct_robohos_enabled", "crude_rate",
     "Robotics – Surgical Systems (%)", "All-Cause Mortality Crude Rate"),

    ("pct_wfaipsn_enabled", "health_outcomes_score",
     "AI – Predict Staff Needs (%)", "Healthcare Quality (CHR Score ↓ better)"),
    ("pct_wfaippd_enabled", "health_outcomes_score",
     "AI – Predict Patient Demand (%)", "Healthcare Quality (CHR Score ↓ better)"),
    ("pct_wfaiss_enabled", "health_outcomes_score",
     "AI – Staff Scheduling (%)", "Healthcare Quality (CHR Score ↓ better)"),
    ("pct_wfaiart_enabled", "health_outcomes_score",
     "AI – Automate Routine Tasks (%)", "Healthcare Quality (CHR Score ↓ better)"),
    ("pct_wfaioacw_enabled", "health_outcomes_score",
     "AI – Optimize Admin./Clinical WF (%)", "Healthcare Quality (CHR Score ↓ better)"),
    ("pct_robohos_enabled", "health_outcomes_score",
     "Robotics – Surgical Systems (%)", "Healthcare Quality (CHR Score ↓ better)"),

    ("pct_wfaipsn_enabled", "clinical_care_score",
     "AI – Predict Staff Needs (%)", "Healthcare Accessibility (CHR Score ↓ better)"),
    ("pct_wfaippd_enabled", "clinical_care_score",
     "AI – Predict Patient Demand (%)", "Healthcare Accessibility (CHR Score ↓ better)"),
    ("pct_wfaiss_enabled", "clinical_care_score",
     "AI – Staff Scheduling (%)", "Healthcare Accessibility (CHR Score ↓ better)"),
    ("pct_wfaiart_enabled", "clinical_care_score",
     "AI – Automate Routine Tasks (%)", "Healthcare Accessibility (CHR Score ↓ better)"),
    ("pct_wfaioacw_enabled", "clinical_care_score",
     "AI – Optimize Admin./Clinical WF (%)", "Healthcare Accessibility (CHR Score ↓ better)"),
    ("pct_robohos_enabled", "clinical_care_score",
     "Robotics – Surgical Systems (%)", "Healthcare Accessibility (CHR Score ↓ better)")
]

# Output folders
BASE_DIR   = Path(__file__).resolve().parent
LOG_DIR    = BASE_DIR / "logs";   LOG_DIR.mkdir(exist_ok=True)
FIG_DIR    = BASE_DIR / "figs";   FIG_DIR.mkdir(exist_ok=True)
OUT_DIR    = BASE_DIR / "output"; OUT_DIR.mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ai_robotics_regressions.log", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Database connection
# --------------------------------------------------------------------------------------
try:
    host     = os.getenv("PGHOST", "localhost")
    db_name  = os.getenv("PGDATABASE", "Research_TEST")
    user     = os.getenv("PGUSER", "postgres")
    password = os.getenv("POSTGRESQL_KEY")  # export POSTGRESQL_KEY=...
    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}/{db_name}"
    engine   = create_engine(conn_str)
    log.info("✔ Connected to Postgres")
except Exception as e:
    log.critical("❌ Could not connect to Postgres: %s", e)
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Pull all required columns in **one** round-trip (cast text→numeric in SQL)
# --------------------------------------------------------------------------------------

# Define which columns come from which table
tech_summary_cols = {
    "county_fips",
    "pct_wfaiart_enabled", 
    "pct_wfaioacw_enabled",
    "pct_wfaippd_enabled", 
    "pct_wfaipsn_enabled",
    "pct_wfaiss_enabled",
    "pct_robohos_enabled",  # Add this new column
    "crude_rate"
}

conceptual_model_cols = {
    "health_outcomes_score",
    "clinical_care_score"
}

# Now construct SQL with correct table references
sql_parts = []

# Add columns from tech_summary table
for col in sorted(tech_summary_cols):
    if col == "county_fips":
        sql_parts.append(f"c.{col}")
    else:
        # Cast to text first before applying regex pattern matching
        sql_parts.append(f"""
            CASE 
                WHEN c.{col}::text ~ '^[-]?[0-9]+(\\.[0-9]+)?$' THEN c.{col}::numeric 
                ELSE NULL 
            END AS {col}
        """.strip())

# Add columns from conceptual_model table
for col in sorted(conceptual_model_cols):
    # Same approach with explicit text cast
    sql_parts.append(f"""
        CASE 
            WHEN m.{col}::text ~ '^[-]?[0-9]+(\\.[0-9]+)?$' THEN m.{col}::numeric 
            ELSE NULL 
        END AS {col}
    """.strip())

# Join the column parts into a complete SQL statement
sql_cols = ",\n    ".join(sql_parts)

query = f"""
SELECT
    {sql_cols}
FROM vw_county_tech_summary_adjpd c
LEFT JOIN vw_conceptual_model_adjpd m
       ON m.county_fips = c.county_fips
-- Exclude malformed state-level FIPS ending with '000'
WHERE c.county_fips NOT LIKE '%%000'
"""

try:
    df = pd.read_sql(text(query), engine)
    log.info("✔ Pulled %d rows of data", len(df))
except Exception as e:
    log.critical("❌ SQL query failed: %s", e)
    sys.exit(1)

# --------------------------------------------------------------------------------------
# Log missing values per column
# --------------------------------------------------------------------------------------
for col in df.columns:
    na_count = df[col].isna().sum()
    if na_count > 0:
        log.info(f"Column {col} has {na_count} missing values ({na_count/len(df):.1%})")

# --------------------------------------------------------------------------------------
# Helper to run one regression & save figure
# --------------------------------------------------------------------------------------
def run_regression(iv, dv, iv_label, dv_label):
    subset = df[[iv, dv]].dropna()
    n = len(subset)
    if n < 30:          # <-- arbitrary threshold; warn if too few obs.
        log.warning("%s vs %s → only %d non-NA rows; results unstable", iv, dv, n)

    # X matrix with constant
    X = sm.add_constant(subset[iv])
    y = subset[dv]

    model = sm.OLS(y, X).fit()
    r2    = model.rsquared
    beta  = model.params[iv]
    pval  = model.pvalues[iv]
    pear_r, pear_p = pearsonr(subset[iv], subset[dv])

    # ── Plot ───────────────────────────────────────────────────────────────
    try:
        # Define the figure path
        fig_path = FIG_DIR / f"{iv}__{dv}.png"
        
        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.figure(figsize=(8, 5))
        
        # Create the regression plot
        ax = sns.regplot(x=iv, y=dv, data=subset,
                         scatter_kws=dict(alpha=0.4, s=25), 
                         line_kws=dict(color='red', linewidth=2))
        
        # Add title and labels
        plt.title(f"Relationship between {iv_label} and {dv_label}", fontsize=14)
        plt.xlabel(iv_label, fontsize=12)
        plt.ylabel(dv_label, fontsize=12)
        
        # Add regression stats in the corner
        plt.text(0.05, 0.95, 
                f"β = {beta:.4f}\nR² = {r2:.3f}\np = {pval:.4g}\nn = {n}", 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Save the figure
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info("    • Saved fig %s", fig_path.name)
    except Exception as e:
        log.error("    × Plot failed for %s vs %s: %s", iv, dv, e)

    return dict(IV=iv, DV=dv, N=n, Beta=beta, p_value=pval,
                R_squared=r2, Pearson_r=pear_r, Pearson_p=pear_p)

# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------
results = []
for iv, dv, iv_lab, dv_lab in IV_DV_PAIRS:
    try:
        log.info("▶ Regression:  %s  →  %s", iv, dv)
        res = run_regression(iv, dv, iv_lab, dv_lab)
        results.append(res)
        log.info("   β=%+.4f  R²=%.3f  p=%g", res["Beta"],
                 res["R_squared"], res["p_value"])
    except Exception as e:
        log.error("   × Failed: %s", e, exc_info=True)

# --------------------------------------------------------------------------------------
# Save summary table
# --------------------------------------------------------------------------------------
summary_df = pd.DataFrame(results).sort_values(["DV", "IV"])
csv_path   = OUT_DIR / "regression_summary.csv"
summary_df.to_csv(csv_path, index=False)
log.info("✔ Wrote summary results to %s", csv_path)

log.info("🎉 All done — see figs/ and output/ for deliverables.")