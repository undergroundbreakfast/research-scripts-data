#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
# -*- coding: utf-8 -*-
"""
Dissertation Replication & Testing Script (v7)
---------------------------------------------------------
This script conducts a comprehensive analysis for a doctoral dissertation focusing 
on the impact of technology adoption on health outcomes and hospital efficiency.

Key Features:
- Connects securely to a PostgreSQL database.
- Fetches and harmonizes data from multiple views and tables based on the provided dictionary.
- Performs robust data preparation, including logging, dummy variable creation, 
  and centering of variables for interaction analysis.
- Runs a series of pre-defined hypothesis tests requested for the dissertation, including:
  - Direct effects using both Clustered OLS and AIPW (for causal inference).
  - Moderation effects using Clustered OLS with interaction terms.
- Includes original script's replication checks and analyses (H1-H4, CAPEX intensity).
- Generates clear, tabular results in CSV format and optional visualizations.
- Produces a single, unified report summarizing all statistical tests.

This is a complete, end-to-end script designed to be run without modification.

Author: Aaron Johnson
"""

import os
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import warnings
import traceback
from typing import Dict, List, Tuple, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection

# -------- Visualization (optional) ----------
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib or seaborn not found. Visualizations will be disabled.")

# =============================================================================
# Logging
# =============================================================================
def setup_logger(log_file_name_prefix="genai_robotics_health_analysis_log"):
    logger = logging.getLogger("genai_robotics_health_analysis")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_file_name_prefix}_{timestamp}.txt"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logging to: {log_path}")

    bootstrap_logger = logging.getLogger("bootstrap_internal")
    if not bootstrap_logger.hasHandlers():
        bootstrap_logger.setLevel(logging.WARNING)
    return logger

# =============================================================================
# DB Connection & Schema Helpers
# =============================================================================
def connect_to_database(logger) -> Engine:
    host = os.getenv("POSTGRES_HOST", 'localhost')
    database = os.getenv("POSTGRES_DB", 'Research_TEST')
    user = os.getenv("POSTGRES_USER", 'postgres')
    password = os.getenv("POSTGRESQL_KEY")

    if password is None:
        logger.error("POSTGRESQL_KEY environment variable not set.")
        sys.exit("Database password not configured. Exiting.")

    try:
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL database '{database}'.")
        return engine
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

def table_exists(engine: Engine, table_name: str, logger) -> bool:
    try:
        with engine.connect() as conn:
            res = conn.execute(
                text("""
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema IN ('public')
                      AND table_name = :tname
                    LIMIT 1
                """),
                {"tname": table_name}
            ).fetchone()
        return res is not None
    except Exception as e:
        logger.warning(f"Table existence check failed for {table_name}: {e}")
        return False

def list_columns(engine: Engine, table_name: str, logger) -> List[str]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = :tname
                """),
                {"tname": table_name}
            ).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        logger.warning(f"Column list failed for {table_name}: {e}")
        return []

def choose_first_existing_table(engine: Engine, candidates: List[str], logger) -> Optional[str]:
    for t in candidates:
        if table_exists(engine, t, logger):
            logger.info(f"Using table/view: {t}")
            return t
    logger.error(f"None of the candidate tables/views exist: {candidates}")
    return None

# =============================================================================
# Data Fetch & Harmonization
# =============================================================================

# Canonical names mapped based on dissertation dictionary's measure descriptions
# Note: Dictionary field names for mo11-mo15 appear mismatched with measures.
# This mapping follows the measure descriptions, which is more reliable.
CANONICAL_TECH_COLS = {
    # MO11: automating routine tasks
    "pct_wfaiart_enabled": "mo11_ai_automate_routine_tasks_pct", 
    # MO12: optimizing...workflows
    "pct_wfaioacw_enabled": "mo12_ai_optimize_workflows_pct",
    # MO13: predicting patient demand
    "pct_wfaippd_enabled": "mo13_ai_predict_patient_demand_pct",
    # MO14: predicting staffing needs
    "pct_wfaipsn_enabled": "mo14_ai_predict_staff_needs_pct",
    # MO15: staff scheduling
    "pct_wfaiss_enabled": "mo15_ai_staff_scheduling_pct",
    # MO21: robotics
    "pct_robohos_enabled": "mo21_robotics_in_hospital_pct",
}

# Column aliases to handle naming variants & _adjpd suffixes.
TECH_COLUMN_ALIASES: Dict[str, List[str]] = {
    "pct_wfaiart_enabled": ["pct_wfaiart_enabled_adjpd", "pct_wfaiart_enabled"],
    "pct_wfaioacw_enabled": ["pct_wfaioacw_enabled_adjpd", "pct_wfaioacw_enabled"],
    "pct_wfaippd_enabled": ["pct_wfaippd_enabled_adjpd", "pct_wfaippd_enabled"],
    "pct_wfaipsn_enabled": ["pct_wfaipsn_enabled_adjpd", "pct_wfaipsn_enabled"],
    "pct_wfaiss_enabled": ["pct_wfaiss_enabled_adjpd", "pct_wfaiss_enabled"],
    "pct_robohos_enabled": ["pct_robohos_enabled_adjpd", "pct_robohos_enabled"],
}

def resolve_first_existing_column(available_cols: List[str], alias_list: List[str]) -> Optional[str]:
    for a in alias_list:
        if a in available_cols:
            return a
    return None

def fetch_data_for_analysis(engine: Engine, logger) -> pd.DataFrame:
    """
    Builds a SELECT based on which tables & columns actually exist. Aligns to canonical column names.
    Adds HRSA rurality (SP5) if available and CAPEX intensity from AHA survey.
    """
    vcm_table = choose_first_existing_table(engine, [
        "vw_conceptual_model_adjpd", "vw_conceptual_model"
    ], logger)
    vcv_table = choose_first_existing_table(engine, [
        "vw_conceptual_model_variables_adjpd", "vw_conceptual_model_variables"
    ], logger)
    vcts_table = choose_first_existing_table(engine, [
        "vw_adjpd_weighted_tech_summary", "vw_county_tech_summary_adjpd", "vw_county_tech_summary"
    ], logger)

    hrsa_table = "hrsa_health_equity_data" if table_exists(engine, "hrsa_health_equity_data", logger) else None
    if hrsa_table:
        logger.info("HRSA health equity table found: hrsa_health_equity_data (for SP5 rurality).")
    else:
        logger.warning("HRSA health equity table NOT found; SP5 rurality models will be skipped if missing.")

    aha_table = "aha_survey_data" if table_exists(engine, "aha_survey_data", logger) else None
    if aha_table:
        logger.info("AHA survey data found: aha_survey_data (for CAPEX intensity).")
    else:
        logger.warning("AHA survey data NOT found; CAPEX intensity models will be skipped.")

    if vcm_table is None:
        sys.exit("Missing conceptual model view (vcm). Aborting.")

    # Columns present in tech table (to resolve aliases)
    tech_cols_available = list_columns(engine, vcts_table, logger) if vcts_table else []
    tech_selects = []
    for canonical_base, canonical_out in CANONICAL_TECH_COLS.items():
        if vcts_table:
            resolved = resolve_first_existing_column(tech_cols_available, TECH_COLUMN_ALIASES[canonical_base])
            if resolved:
                tech_selects.append(f"vcts.{resolved} AS {canonical_out}")
            else:
                tech_selects.append(f"NULL::numeric AS {canonical_out}")
                logger.warning(f"Tech column not found for {canonical_out}. Filled with NULL.")
        else:
            tech_selects.append(f"NULL::numeric AS {canonical_out}")

    # Base SELECT (index-level variables & outcomes)
    sql_parts = [f"""
        SELECT
            vcm.county_fips,
            vcm.health_behaviors_score             AS iv3_health_behaviors_score,
            vcm.social_economic_factors_score      AS iv4_social_economic_factors_score,
            vcm.physical_environment_score         AS iv2_physical_environment_score,
            vcm.medicaid_expansion_active          AS iv1_medicaid_expansion_active,

            vcm.health_outcomes_score              AS dv2_health_outcomes_score,
            vcm.clinical_care_score                AS dv1_clinical_care_score,
            vcm.avg_patient_services_margin        AS dv3_avg_patient_services_margin,

            vcm.population                         AS population,
            vcm.census_division                    AS census_division,

            vcm.weighted_ai_adoption_score         AS mo1_genai_composite_score,
            vcm.weighted_robotics_adoption_score   AS mo2_robotics_composite_score
    """]

    # DV components (optional)
    if vcv_table:
        sql_parts.append(f"""
            , vcv.premature_death_raw_value                AS dv21_premature_death_ypll_rate
            , vcv.ratio_of_population_to_primary_care_physicians AS dv12_physicians_ratio
            , vcv.preventable_hospital_stays_raw_value     AS dv15_preventable_stays_rate
        """)
    else:
        sql_parts.append("""
            , NULL::numeric AS dv21_premature_death_ypll_rate
            , NULL::numeric AS dv12_physicians_ratio
            , NULL::numeric AS dv15_preventable_stays_rate
        """)

    # SP5 rurality (HRSA IRR)
    if hrsa_table:
        sql_parts.append("""
            , hrsa.irr_county_value::numeric AS sp5_irr_county_value
        """)
    else:
        sql_parts.append(", NULL::numeric AS sp5_irr_county_value")

    # CAPEX intensity terms
    if aha_table:
        sql_parts.append("""
            , aha.capex_sum::numeric AS fi1_capex_sum
            , aha.adjpd_sum::numeric AS fi2_adjpd_sum
            , CASE WHEN aha.adjpd_sum IS NOT NULL AND aha.adjpd_sum <> 0
                   THEN (aha.capex_sum::numeric / aha.adjpd_sum::numeric)
                   ELSE NULL END AS fi_capex_intensity_ratio
        """)
    else:
        sql_parts.append("""
            , NULL::numeric AS fi1_capex_sum
            , NULL::numeric AS fi2_adjpd_sum
            , NULL::numeric AS fi_capex_intensity_ratio
        """)

    # Tech component columns
    if vcts_table:
        sql_parts.append(", " + ", ".join(tech_selects))

    # FROM / JOINs
    sql_parts.append(f"FROM public.{vcm_table} AS vcm")
    if vcv_table:
        sql_parts.append(f"LEFT JOIN public.{vcv_table}  AS vcv  ON vcm.county_fips = vcv.county_fips")
    if vcts_table:
        sql_parts.append(f"LEFT JOIN public.{vcts_table} AS vcts ON vcm.county_fips = vcts.county_fips")
    if hrsa_table:
        # Normalize FIPS in HRSA (pad to 5), aggregate in case of duplicates
        sql_parts.append(f"""
            LEFT JOIN (
                SELECT
                    LPAD(TRIM(CAST(county_fips_code AS TEXT)), 5, '0') AS county_fips,
                    AVG(CASE
                          WHEN CAST(NULLIF(TRIM(CAST(irr_county_value AS TEXT)), '') AS TEXT) ~ '^[0-9]+(\\.[0-9]+)?$'
                          THEN irr_county_value::numeric
                          ELSE NULL
                        END) AS irr_county_value
                FROM public.{hrsa_table}
                GROUP BY 1
            ) AS hrsa
              ON vcm.county_fips = hrsa.county_fips
        """)

    if aha_table:
        # Aggregate AHA by county FIPS with numeric safety & padding (fcounty)
        sql_parts.append(f"""
            LEFT JOIN (
                SELECT
                    LPAD(TRIM(CAST(fcounty AS TEXT)), 5, '0') AS county_fips,
                    SUM(CASE WHEN CAST(NULLIF(TRIM(CAST(ceamt AS TEXT)), '') AS TEXT) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                             THEN ceamt::numeric ELSE NULL END) AS capex_sum,
                    SUM(CASE WHEN CAST(NULLIF(TRIM(CAST(adjpd AS TEXT)), '') AS TEXT) ~ '^-?[0-9]+(\\.[0-9]+)?$'
                             THEN adjpd::numeric ELSE NULL END) AS adjpd_sum
                FROM public.{aha_table}
                GROUP BY 1
            ) AS aha
              ON vcm.county_fips = aha.county_fips
        """)

    sql_parts.append("WHERE vcm.population IS NOT NULL AND CAST(vcm.population AS NUMERIC) > 0;")
    sql_query = "\n".join(sql_parts)

    try:
        df = pd.read_sql_query(text(sql_query), engine)
        if df.empty:
            raise RuntimeError("Fetched DataFrame is empty. Check views/joins/filters.")
        # FIPS hygiene
        df['county_fips'] = df['county_fips'].astype(str).str.zfill(5)
        before = len(df)
        df = df[~df['county_fips'].str.endswith("000")]
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with FIPS ending in '000' (non-county).")

        # Numeric coercions
        numeric_cols = [
            'population', 'iv3_health_behaviors_score', 'iv4_social_economic_factors_score',
            'iv2_physical_environment_score', 'dv2_health_outcomes_score',
            'dv1_clinical_care_score', 'dv3_avg_patient_services_margin',
            'mo1_genai_composite_score', 'mo2_robotics_composite_score',
            'dv21_premature_death_ypll_rate', 'dv15_preventable_stays_rate', 
            'sp5_irr_county_value', 'fi1_capex_sum', 'fi2_adjpd_sum', 'fi_capex_intensity_ratio'
        ] + list(CANONICAL_TECH_COLS.values())
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Keep positive population
        df = df[df['population'].notna() & (df['population'] > 0)]
        logger.info(f"Data retrieved: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

# =============================================================================
# Common preparation
# =============================================================================
def common_prepare_data(df_input: pd.DataFrame, logger) -> Tuple[pd.DataFrame, List[str]]:
    df = df_input.copy()
    df.columns = [c.lower() for c in df.columns]

    # Cluster col: state FIPS
    if 'county_fips' not in df.columns:
        logger.error("county_fips not found after fetch. Exiting.")
        sys.exit(1)
    df['state_fips_for_clustering'] = df['county_fips'].astype(str).str[:2]

    # Log(pop)
    if 'population' not in df.columns:
        logger.error("population not found after fetch. Exiting.")
        sys.exit(1)
    # Filter out non-positive population before log
    df = df[df['population'] > 0]
    df['log_population'] = np.log(df['population'])

    # Medicaid expansion to numeric 0/1
    medicaid_col = 'iv1_medicaid_expansion_active'
    if medicaid_col in df.columns:
        if df[medicaid_col].dtype == bool or df[medicaid_col].dtype == np.bool_:
            df[medicaid_col] = df[medicaid_col].astype(int)
        else:
            df[medicaid_col] = (df[medicaid_col].astype(str).str.lower()
                                .map({'true':1, 't':1, 'yes':1, 'y':1, '1':1,
                                      'false':0, 'f':0, 'no':0, 'n':0, '0':0}))
        df[medicaid_col] = pd.to_numeric(df[medicaid_col], errors='coerce')
    else:
        logger.warning("Medicaid expansion column missing; moderation H9 will skip.")

    # Census Division dummies
    census_dummy_cols = []
    if 'census_division' in df.columns:
        df['census_division'] = df['census_division'].astype(str)
        dummies = pd.get_dummies(df['census_division'], prefix='div', drop_first=True, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        census_dummy_cols = list(dummies.columns)
        logger.info(f"Created {len(census_dummy_cols)} division dummies.")
    else:
        logger.warning("census_division missing; proceeding without division dummies.")

    base_controls = [
        'iv4_social_economic_factors_score',
        'iv2_physical_environment_score',
        'iv3_health_behaviors_score',
        'iv1_medicaid_expansion_active',
        'log_population'
    ] + census_dummy_cols
    base_controls = [c for c in base_controls if c in df.columns]

    # Ensure numeric for outcome/exposure columns if present
    for col in [
        'dv21_premature_death_ypll_rate', 'dv2_health_outcomes_score',
        'dv15_preventable_stays_rate', 'dv1_clinical_care_score',
        'dv3_avg_patient_services_margin',
        'mo1_genai_composite_score', 'mo2_robotics_composite_score',
        'sp5_irr_county_value',
        'fi1_capex_sum', 'fi2_adjpd_sum', 'fi_capex_intensity_ratio'
    ] + list(CANONICAL_TECH_COLS.values()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs for tech adoption components with 0 (missing=not adopted)
    for tcol in list(CANONICAL_TECH_COLS.values()):
        if tcol in df.columns and df[tcol].isna().any():
            n = df[tcol].isna().sum()
            df[tcol] = df[tcol].fillna(0)
            logger.info(f"Filled {n} NaNs with 0 for tech column '{tcol}'.")
    if 'mo1_genai_composite_score' in df.columns and df['mo1_genai_composite_score'].isna().any():
        n = df['mo1_genai_composite_score'].isna().sum()
        df['mo1_genai_composite_score'] = df['mo1_genai_composite_score'].fillna(0)
        logger.info(f"Filled {n} NaNs with 0 for 'mo1_genai_composite_score'.")

    # Center variables used in interactions to reduce collinearity
    def center_series(s: pd.Series) -> pd.Series:
        return s - s.mean()

    vars_to_center = [
        'mo1_genai_composite_score', 'mo2_robotics_composite_score',
        'iv2_physical_environment_score', 'iv3_health_behaviors_score', 
        'sp5_irr_county_value'
    ] + list(CANONICAL_TECH_COLS.values())
    
    for col in vars_to_center:
        if col in df.columns:
            df[f"{col}_c"] = center_series(pd.to_numeric(df[col], errors='coerce'))
            logger.info(f"Created centered variable: {col}_c")

    # CAPEX winsorization & log transforms
    if 'fi_capex_intensity_ratio' in df.columns:
        s = df['fi_capex_intensity_ratio']
        if s.notna().sum() > 10:
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            df['fi_capex_intensity_ratio_w'] = s.clip(lower=lo, upper=hi)
            df['fi_capex_intensity_ratio_log1p'] = np.log1p(df['fi_capex_intensity_ratio_w'])
            logger.info("Constructed CAPEX intensity winsorized and log1p variables.")
        else:
            df['fi_capex_intensity_ratio_w'] = s
            df['fi_capex_intensity_ratio_log1p'] = np.log1p(s.replace({-np.inf: np.nan, np.inf: np.nan}))
    else:
        logger.warning("No CAPEX intensity found; CAPEX models will be skipped.")

    return df, base_controls

# =============================================================================
# Modeling helpers
# =============================================================================
def add_const(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant='add')

def run_ols_clustered(Y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(Y, add_const(X))
    if clusters.nunique() >= 2:
        res = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})
    else:
        res = model.fit()
    return res

def backward_stepwise_by_p(Y: pd.Series, X: pd.DataFrame, keep: List[str], p_remove: float = 0.10) -> List[str]:
    vars_in = list(X.columns)
    changed = True
    while changed and len(vars_in) > len(keep):
        changed = False
        model = sm.OLS(Y, add_const(X[vars_in])).fit()  # plain OLS for selection
        pvals = model.pvalues.drop('const', errors='ignore')
        pvals = pvals.drop([v for v in keep if v in pvals.index], errors='ignore')
        if not pvals.empty:
            worst_var = pvals.idxmax()
            if pvals[worst_var] > p_remove and worst_var not in keep:
                vars_in.remove(worst_var)
                changed = True
    return vars_in

def _bh_correct_in_place(df, p_col, group_col, q_col_out):
    if df.empty or p_col not in df.columns:
        df[q_col_out] = np.nan
        return df
    if group_col is None:
        mask = df[p_col].notna()
        df[q_col_out] = np.nan
        if mask.any():
            rej, q = fdrcorrection(df.loc[mask, p_col].values, alpha=0.05, method='indep')
            df.loc[mask, q_col_out] = q
        return df
    df[q_col_out] = np.nan
    for g, dfg in df.groupby(group_col):
        mask = dfg[p_col].notna()
        if mask.any():
            rej, q = fdrcorrection(dfg.loc[mask, p_col].values, alpha=0.05, method='indep')
            df.loc[dfg.index[mask], q_col_out] = q
    return df

def _coef_se(res, name):
    try:
        return float(res.bse[name])
    except Exception:
        return np.nan

# =============================================================================
# IPTW / AIPW
# =============================================================================
def estimate_propensity_scores(X_confounders, T_treatment, logger, treatment_name, C_param=0.1):
    scaler = StandardScaler()
    Xn = X_confounders.apply(pd.to_numeric, errors='coerce').copy()
    if Xn.isnull().any().any():
        Xn = Xn.fillna(Xn.mean())
    Xs = scaler.fit_transform(Xn)
    try:
        if len(np.unique(T_treatment)) < 2:
            logger.error(f"PS model ({treatment_name}): treatment has one class.")
            return None, None
        if np.min(np.bincount(T_treatment.astype(int))) < 5:
            logger.warning(f"PS model ({treatment_name}): class imbalance; results may be unstable.")
        lr = LogisticRegression(solver='liblinear', random_state=42, C=C_param, penalty='l1', max_iter=300)
        lr.fit(Xs, T_treatment)
        ps = lr.predict_proba(Xs)[:, 1]
    except Exception as e:
        logger.error(f"PS model fitting failed for {treatment_name}: {e}")
        return None, None
    ps = np.clip(ps, 0.01, 0.99)
    return ps, scaler

def run_aipw(df, treat_col, y_col, confounders, n_boot, logger, plot_dir):
    T = df[treat_col].astype(int).values
    Y = df[y_col].values
    X = df[confounders].copy()
    if X.isnull().any().any():
        X = X.fillna(X.mean())

    ps, scaler = estimate_propensity_scores(X, T, logger, treat_col)
    if ps is None or scaler is None:
        return (np.nan, np.nan, np.nan, np.nan,
                int((T == 1).sum()), int((T == 0).sum()), True)

    # PS overlap plot
    if VISUALIZATION_AVAILABLE:
        plt.figure(figsize=(9, 5))
        try:
            sns.histplot(ps[T == 1], label=f"Treated (N={(T==1).sum()})", stat="density", kde=True, bins=30, alpha=0.6)
            sns.histplot(ps[T == 0], label=f"Control (N={(T==0).sum()})", stat="density", kde=True, bins=30, alpha=0.6)
            plt.title(f"Propensity Score Overlap: {treat_col}")
            plt.xlabel("Propensity score"); plt.legend(); plt.tight_layout()
            os.makedirs(plot_dir, exist_ok=True)
            outp = os.path.join(plot_dir, f"ps_overlap_{treat_col.replace('%','pct')}.png")
            plt.savefig(outp, dpi=300); plt.close()
        except Exception:
            plt.close()
    
    def _fit_mu_single(X_all, Y_all, mask):
        y = Y_all[mask]
        Xa = X_all.copy()
        if np.isnan(Xa).any():
            col_means = np.nanmean(Xa, axis=0)
            inds = np.where(np.isnan(Xa))
            Xa[inds] = np.take(col_means, inds[1])
        if mask.sum() > Xa.shape[1] and mask.sum() > 5:
            try:
                m = sm.OLS(y, add_const(Xa[mask])).fit()
                return m.predict(add_const(Xa))
            except Exception:
                return np.full(len(Y_all), np.mean(y) if mask.sum() > 0 else np.mean(Y_all))
        else:
            return np.full(len(Y_all), np.mean(y) if mask.sum() > 0 else np.mean(Y_all))

    Xs = scaler.transform(X.apply(pd.to_numeric, errors='coerce'))
    mu1 = _fit_mu_single(Xs, Y, (T == 1))
    mu0 = _fit_mu_single(Xs, Y, (T == 0))
    
    term1 = (T / ps) * (Y - mu1) + mu1
    term0 = ((1 - T) / (1 - ps)) * (Y - mu0) + mu0
    ate = np.mean(term1[np.isfinite(term1)]) - np.mean(term0[np.isfinite(term0)])

    # Bootstrap CIs and p-approx
    ate_boot = []
    df_reset = df.reset_index(drop=True)
    for i in range(n_boot):
        try:
            b = resample(df_reset, replace=True, random_state=i)
            if b.empty or b[treat_col].nunique() < 2:
                continue
            T_b = b[treat_col].astype(int).values
            Y_b = b[y_col].values
            X_b = b[confounders].copy().fillna(b[confounders].mean())
            ps_b, sc_b = estimate_propensity_scores(X_b, T_b, logging.getLogger("bootstrap_internal"), treat_col)
            if ps_b is None or sc_b is None:
                continue
            Xs_b = sc_b.transform(X_b.apply(pd.to_numeric, errors='coerce'))
            mu1_b = _fit_mu_single(Xs_b, Y_b, (T_b == 1))
            mu0_b = _fit_mu_single(Xs_b, Y_b, (T_b == 0))
            
            term1_b = (T_b / ps_b) * (Y_b - mu1_b) + mu1_b
            term0_b = ((1 - T_b) / (1 - ps_b)) * (Y_b - mu0_b) + mu0_b
            ate_b = np.mean(term1_b[np.isfinite(term1_b)]) - np.mean(term0_b[np.isfinite(term0_b)])

            if np.isfinite(ate_b): ate_boot.append(ate_b)
        except Exception:
            pass

    ate_ci = (np.percentile(ate_boot, 2.5), np.percentile(ate_boot, 97.5)) if ate_boot else (np.nan, np.nan)

    # two-sided bootstrap p approx (centered at 0)
    p_raw = np.nan
    if ate_boot and np.isfinite(ate):
        boots = np.array(ate_boot)
        if ate > 0:
            p_raw = 2 * (np.sum(boots <= 0) + 1) / (len(boots) + 1)
        elif ate < 0:
            p_raw = 2 * (np.sum(boots >= 0) + 1) / (len(boots) + 1)
        else:
            p_raw = 1.0
        p_raw = float(min(p_raw, 1.0))

    return (ate, ate_ci[0], ate_ci[1], p_raw,
            int((T == 1).sum()), int((T == 0).sum()), False)

# =============================================================================
# E-value & Lives-saved sim
# =============================================================================
def calculate_e_value(effect, sd_ctrl):
    if np.isnan(effect) or np.isnan(sd_ctrl) or sd_ctrl == 0:
        return np.nan
    smd = abs(effect) / sd_ctrl
    if smd == 0:
        return 1.0
    rr_approx = np.exp(0.91 * smd)
    if rr_approx <= 1:
        return 1.0
    try:
        return rr_approx + np.sqrt(rr_approx * (rr_approx - 1))
    except Exception:
        return np.nan

# =============================================================================
# Plots
# =============================================================================
def plot_interaction_continuous(df: pd.DataFrame, A_col: str, B_col: str, Y_col: str,
                                name: str, beta: float, pval: float, plot_dir: str, logger):
    if not VISUALIZATION_AVAILABLE or any(c not in df.columns for c in [A_col, B_col, Y_col]):
        return
    d = df[[A_col, B_col, Y_col]].dropna().copy()
    if len(d) < 20:
        return
    try:
        from sklearn.linear_model import LinearRegression
        d['_AxB'] = d[A_col] * d[B_col]
        X = d[[A_col, B_col, '_AxB']].values
        y = d[Y_col].values
        lr = LinearRegression().fit(X, y)
        b_vals = [d[B_col].quantile(0.25), d[B_col].quantile(0.75)]
        a_min, a_max = d[A_col].quantile(0.02), d[A_col].quantile(0.98)
        a_grid = np.linspace(a_min, a_max, 100)
        plt.figure(figsize=(9, 6))
        plt.scatter(d[A_col], d[Y_col], alpha=0.25, s=18, label="Data points")
        colors = ['blue', 'red']
        for i, b in enumerate(b_vals):
            Xline = np.column_stack([a_grid, np.full_like(a_grid, b), a_grid*b])
            yhat = lr.predict(Xline)
            plt.plot(a_grid, yhat, linewidth=2.5, color=colors[i], label=f"Moderator at {b:.2f} (25/75th %ile)")
        plt.title(f"{name} | β_int={beta:.2f}, p={pval:.3f}")
        plt.xlabel(A_col.replace('_c','').replace('_',' ').title())
        plt.ylabel(Y_col.replace('_',' ').title())
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        outp = os.path.join(plot_dir, f"interaction_cont_{name}.png".replace(" ", "_"))
        plt.savefig(outp, dpi=300)
        plt.close()
        logger.info(f"Saved continuous interaction plot: {outp}")
    except Exception as e:
        logger.error(f"Error creating continuous interaction plot for {name}: {e}")
        plt.close()

def assemble_replication_report_full(out_dir,
                                     idx_df=None, mod_df=None, main_df=None,
                                     rural_df=None, dv21_rural_df=None,
                                     strat_df=None, h1h4_df=None,
                                     capex_df=None, dv3_df=None, 
                                     dissertation_direct_df=None,
                                     dissertation_interaction_df=None,
                                     logger=None):
    rows = []

    def add_row(fam, test, N, beta, p, lo, hi, q=None, note=""):
        rows.append({"Family": fam, "Test": test, "N": N,
                     "Beta": beta, "p": p, "CI_lower": lo, "CI_upper": hi,
                     "Q_FDR": q, "Note": note})

    # Dissertation Direct Effects
    if dissertation_direct_df is not None and not dissertation_direct_df.empty:
        for _, r in dissertation_direct_df.iterrows():
            # Report AIPW if available, otherwise OLS
            if pd.notna(r.get("AIPW_ATE")):
                beta, p, lo, hi = r["AIPW_ATE"], r["AIPW_p"], r["AIPW_CI_Lower"], r["AIPW_CI_Upper"]
                note = f"AIPW (N_t={int(r.get('N_Treated', 0))}, N_c={int(r.get('N_Control', 0))})"
            else:
                beta, p, lo, hi = r["OLS_Beta"], r["OLS_p"], r["OLS_CI_Lower"], r["OLS_CI_Upper"]
                note = "OLS (continuous)"
            add_row("Dissertation Direct Effects",
                    f"{r['Treatment']} -> {r['Outcome']}",
                    r.get("N"), beta, p, lo, hi, note=note)

    # Dissertation Interaction Effects
    if dissertation_interaction_df is not None and not dissertation_interaction_df.empty:
        for _, r in dissertation_interaction_df.iterrows():
            add_row("Dissertation Interaction Effects",
                    f"{r['IV']} x {r['Moderator']} -> {r['Outcome']}",
                    r.get("N"), r.get("Interaction_Beta"), r.get("Interaction_p"),
                    r.get("Interaction_CI_Lower"), r.get("Interaction_CI_Upper"))

    # Replication (MO1/MO2 → DVs)
    if idx_df is not None and not idx_df.empty:
        s = idx_df[idx_df["Exposure_Col"].isin(["mo1_genai_composite_score","mo2_robotics_composite_score"])]
        for _, r in s.iterrows():
            add_row("Replication (Index OLS)",
                    f"{r['Exposure_Col']} -> {r['Outcome_Col']}",
                    r.get("N"), r.get("OLS Beta"), r.get("OLS p"),
                    r.get("CI_Lower"), r.get("CI_Upper"), None, r.get("Exposure"))

    # H1–H4 results
    if h1h4_df is not None and not h1h4_df.empty:
        for _, r in h1h4_df.iterrows():
            add_row("H1–H4",
                    f"{r['Hypothesis']}: {r['Exposure']} -> {r['Outcome']}",
                    r.get("N"), r.get("OLS Beta"), r.get("OLS p"),
                    r.get("CI_Lower"), r.get("CI_Upper"), r.get("BH Q-Value"))

    # CAPEX intensity
    if capex_df is not None and not capex_df.empty:
        for _, r in capex_df.iterrows():
            add_row("CAPEX intensity (FI1/FI2)",
                    f"{r['Predictor']} -> {r['Outcome']}",
                    r.get("N"), r.get("OLS Beta"), r.get("OLS p"),
                    r.get("CI_Lower"), r.get("CI_Upper"), r.get("BH Q-Value"), r["Predictor_Col"])

    out_csv = os.path.join(out_dir, "replication_report_full.csv")
    pd.DataFrame(rows).sort_values("Family").to_csv(out_csv, index=False)
    if logger: logger.info(f"Wrote unified replication report to {out_csv} (rows={len(rows)})")

# =============================================================================
# H1–H4 Hypothesis Tests (IVs → DVs)
# =============================================================================
def run_h1_h4_tests(df: pd.DataFrame, base_controls: List[str], logger, out_dir: str) -> pd.DataFrame:
    """
    Runs clustered-OLS per hypothesis. Controls include log(pop) + division dummies + other IVs (excluding the IV under test).
    """
    specs = [
        # H1
        {"H":"H1", "Exposure":"iv1_medicaid_expansion_active", "Outcome":"dv1_clinical_care_score"},
        {"H":"H1", "Exposure":"iv1_medicaid_expansion_active", "Outcome":"dv3_avg_patient_services_margin"},
        # H2
        {"H":"H2", "Exposure":"iv2_physical_environment_score", "Outcome":"dv1_clinical_care_score"},
        {"H":"H2", "Exposure":"iv2_physical_environment_score", "Outcome":"dv2_health_outcomes_score"},
        {"H":"H2", "Exposure":"iv2_physical_environment_score", "Outcome":"dv3_avg_patient_services_margin"},
        # H3
        {"H":"H3", "Exposure":"iv3_health_behaviors_score", "Outcome":"dv1_clinical_care_score"},
        {"H":"H3", "Exposure":"iv3_health_behaviors_score", "Outcome":"dv2_health_outcomes_score"},
        # H4
        {"H":"H4", "Exposure":"iv4_social_economic_factors_score", "Outcome":"dv1_clinical_care_score"},
        {"H":"H4", "Exposure":"iv4_social_economic_factors_score", "Outcome":"dv2_health_outcomes_score"},
        {"H":"H4", "Exposure":"iv4_social_economic_factors_score", "Outcome":"dv3_avg_patient_services_margin"},
    ]

    rows = []
    for s in specs:
        xcol, ycol = s["Exposure"], s["Outcome"]
        missing = [c for c in [xcol, ycol, 'state_fips_for_clustering'] if c not in df.columns]
        if missing:
            logger.warning(f"Skipping {s['H']} {xcol}->{ycol}: missing {missing}")
            continue

        # Controls = base_controls minus the IV under test
        controls = [c for c in base_controls if c != xcol]
        cols = [ycol, xcol, 'state_fips_for_clustering'] + controls
        d = df[cols].dropna()
        if d.empty:
            logger.warning(f"{s['H']} {xcol}->{ycol}: no analytical rows after dropna.")
            continue

        Y = d[ycol]
        X = d[[xcol] + controls]
        res = run_ols_clustered(Y, X, d['state_fips_for_clustering'])
        b = res.params.get(xcol, np.nan)
        p = res.pvalues.get(xcol, np.nan)
        ci = res.conf_int().loc[xcol].values if xcol in res.params.index else (np.nan, np.nan)

        rows.append({
            "Hypothesis": s["H"], "Exposure": xcol, "Outcome": ycol,
            "N": len(d), "OLS Beta": b, "OLS p": p, "CI_Lower": ci[0], "CI_Upper": ci[1]
        })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = _bh_correct_in_place(out_df, 'OLS p', None, 'BH Q-Value')
        out_csv = os.path.join(out_dir, "hypotheses_h1_h4_summary.csv")
        out_df.to_csv(out_csv, index=False)
        logger.info(f"Saved H1–H4 summary to {out_csv}")
    else:
        logger.warning("No H1–H4 results to save.")

    return out_df

# =============================================================================
# CAPEX intensity (FI1/FI2) → DV2, DV21, MO1, MO2
# =============================================================================
def run_capex_intensity_tests(df: pd.DataFrame, base_controls: List[str], logger, out_dir: str) -> pd.DataFrame:
    if 'fi_capex_intensity_ratio' not in df.columns:
        logger.warning("CAPEX intensity not found; skipping CAPEX models.")
        return pd.DataFrame()

    outcomes = [
        ("dv2_health_outcomes_score", "DV2_Health_Outcomes"),
        ("dv21_premature_death_ypll_rate", "DV21_YPLL"),
        ("mo1_genai_composite_score", "MO1_GenAI"),
        ("mo2_robotics_composite_score", "MO2_Robotics"),
    ]

    xvars = [
        ("fi_capex_intensity_ratio", "CAPEX_Intensity_Raw"),
        ("fi_capex_intensity_ratio_w", "CAPEX_Intensity_Winsor01_99"),
        ("fi_capex_intensity_ratio_log1p", "log1p_CAPEX_Intensity_Winsor"),
    ]

    rows = []
    for ycol, yname in outcomes:
        if ycol not in df.columns:
            logger.warning(f"CAPEX: outcome {ycol} missing; skipping.")
            continue
        for xcol, xname in xvars:
            if xcol not in df.columns:
                continue
            controls = [c for c in base_controls if c != xcol]
            cols = [ycol, xcol, 'state_fips_for_clustering'] + controls
            d = df[cols].dropna()
            if d.empty or d[xcol].nunique() <= 1:
                continue

            Y = d[ycol]
            X = d[[xcol] + controls]
            try:
                res = run_ols_clustered(Y, X, d['state_fips_for_clustering'])
                b = res.params.get(xcol, np.nan)
                p = res.pvalues.get(xcol, np.nan)
                ci = res.conf_int().loc[xcol].values if xcol in res.params.index else (np.nan, np.nan)
            except Exception:
                b, p, ci = np.nan, np.nan, (np.nan, np.nan)

            rows.append({
                "Outcome": yname, "Outcome_Col": ycol,
                "Predictor": xname, "Predictor_Col": xcol,
                "N": len(d),
                "OLS Beta": b, "OLS p": p,
                "CI_Lower": ci[0], "CI_Upper": ci[1]
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = _bh_correct_in_place(out, 'OLS p', 'Outcome', 'BH Q-Value')
        out_csv = os.path.join(out_dir, "capex_intensity_regression_summary.csv")
        out.to_csv(out_csv, index=False)
        logger.info(f"Saved CAPEX intensity regression summary to {out_csv}")
    else:
        logger.warning("No CAPEX intensity results to save.")
    return out

# =============================================================================
# MAIN
# =============================================================================
def main():
    logger = setup_logger()
    logger.info("Starting Dissertation Replication & Testing Script.")
    # Directories
    out_dir = "genai_robotics_health_output"
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.PerfectSeparationWarning)

    # Parameters
    N_BOOT = 500
    
    # Connect & fetch
    engine = connect_to_database(logger)
    df_raw = fetch_data_for_analysis(engine, logger)
    df, base_controls = common_prepare_data(df_raw, logger)

    # --------------------------------------------------------------------------------
    # PART A-G: Original script's analyses are run first
    # --------------------------------------------------------------------------------
    logger.info("Running original script analyses (Parts A-G).")
    
    # PART C. Index-level OLS (for replication checks)
    outcomes_c = {
        "dv1_clinical_care_score": "DV1_Clinical_Care",
        "dv2_health_outcomes_score": "DV2_Health_Outcomes",
        "dv3_avg_patient_services_margin": "DV3_Patient_Services_Margin"
    }
    exposures_c = {
        "mo1_genai_composite_score": "MO1_GenAI_Composite",
        "mo2_robotics_composite_score": "MO2_Robotics_Composite"
    }
    
    idx_rows = []
    for xcol, xname in exposures_c.items():
        if xcol not in df.columns: continue
        for ycol, yname in outcomes_c.items():
            if ycol not in df.columns: continue
            d = df[[ycol, xcol, 'state_fips_for_clustering'] + base_controls].dropna()
            if d.empty: continue
            Y = d[ycol]; X = d[[xcol] + base_controls]
            res = run_ols_clustered(Y, X, d['state_fips_for_clustering'])
            b = res.params.get(xcol, np.nan); p = res.pvalues.get(xcol, np.nan)
            ci = res.conf_int().loc[xcol].values if xcol in res.params.index else (np.nan, np.nan)
            idx_rows.append({
                "Exposure": xname, "Exposure_Col": xcol, "Outcome": yname, "Outcome_Col": ycol,
                "N": len(d), "OLS Beta": b, "OLS p": p, "CI_Lower": ci[0], "CI_Upper": ci[1]
            })

    idx_df = pd.DataFrame(idx_rows)
    if not idx_df.empty: idx_df.to_csv(os.path.join(out_dir, "index_level_ols_summary.csv"), index=False)
    
    # Other analyses from original script
    h1h4_df = run_h1_h4_tests(df, base_controls, logger, out_dir)
    capex_df = run_capex_intensity_tests(df, base_controls, logger, out_dir)
    
    # ================================================================================
    # PART H: DISSERTATION HYPOTHESIS TESTS
    # ================================================================================
    logger.info("PART H: Running all dissertation-specific hypothesis tests.")
    
    # --- H1. Direct Effects ---
    # This list combines tests from the original script AND newly requested tests.
    direct_effects_specs = [
        # --- Tests from original script (preserved) ---
        {"treatment": "mo1_genai_composite_score", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo11_ai_automate_routine_tasks_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo12_ai_optimize_workflows_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo13_ai_predict_patient_demand_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo14_ai_predict_staff_needs_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "mo15_ai_staff_scheduling_pct", "outcome": "dv15_preventable_stays_rate"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo1_genai_composite_score"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo11_ai_automate_routine_tasks_pct"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo12_ai_optimize_workflows_pct"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo13_ai_predict_patient_demand_pct"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo14_ai_predict_staff_needs_pct"},
        {"treatment": "sp5_irr_county_value", "outcome": "mo15_ai_staff_scheduling_pct"},
        # --- Newly requested tests (added) ---
        {"treatment": "mo2_robotics_composite_score", "outcome": "dv15_preventable_stays_rate"}, # MO2-->DV15
        {"treatment": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},  # MO21-->DV15
    ]
    
    direct_results = []
    for spec in direct_effects_specs:
        t_col, y_col = spec["treatment"], spec["outcome"]
        logger.info(f"  Direct effect test: {t_col} -> {y_col}")
        
        required_cols = [t_col, y_col, 'state_fips_for_clustering'] + base_controls
        if any(c not in df.columns for c in required_cols):
            logger.warning(f"    Skipping: missing one or more columns from: {required_cols}")
            continue
            
        d = df[required_cols].dropna()
        if d.empty or d[t_col].nunique() < 2:
            logger.warning("    Skipping: Not enough data or variation after dropping NaNs.")
            continue
            
        # OLS with continuous treatment
        res_ols = run_ols_clustered(d[y_col], d[[t_col] + base_controls], d['state_fips_for_clustering'])
        ols_beta = res_ols.params.get(t_col, np.nan)
        ols_p = res_ols.pvalues.get(t_col, np.nan)
        ols_ci = res_ols.conf_int().loc[t_col].values if t_col in res_ols.params.index else (np.nan, np.nan)
        
        # AIPW with binarized treatment
        median_val = d[t_col].median()
        bin_col = f"{t_col}_gt_median"
        d[bin_col] = (d[t_col] > median_val).astype(int)

        if d[bin_col].nunique() < 2:
             aipw_ate, aipw_cl, aipw_cu, aipw_p, n_t, n_c = [np.nan] * 6
        else:
            (aipw_ate, aipw_cl, aipw_cu, aipw_p,
            n_t, n_c, aipw_err) = run_aipw(d, bin_col, y_col, base_controls, N_BOOT, logger, plot_dir)

        direct_results.append({
            "Treatment": t_col, "Outcome": y_col, "N": len(d),
            "OLS_Beta": ols_beta, "OLS_p": ols_p, "OLS_CI_Lower": ols_ci[0], "OLS_CI_Upper": ols_ci[1],
            "AIPW_ATE": aipw_ate, "AIPW_p": aipw_p, "AIPW_CI_Lower": aipw_cl, "AIPW_CI_Upper": aipw_cu,
            "N_Treated": n_t, "N_Control": n_c,
        })
    
    dissertation_direct_df = pd.DataFrame(direct_results)
    if not dissertation_direct_df.empty:
        out_csv = os.path.join(out_dir, "dissertation_direct_effects_summary.csv")
        dissertation_direct_df.to_csv(out_csv, index=False)
        logger.info(f"Saved dissertation direct effects summary to {out_csv}")

    # --- H2. Interaction Effects ---
    # This list combines tests from the original script AND newly requested tests.
    interaction_specs = [
        # --- Tests from original script (preserved) ---
        {"iv": "iv3_health_behaviors_score", "moderator": "mo1_genai_composite_score", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo11_ai_automate_routine_tasks_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo12_ai_optimize_workflows_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo13_ai_predict_patient_demand_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo14_ai_predict_staff_needs_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv3_health_behaviors_score", "moderator": "mo15_ai_staff_scheduling_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo1_genai_composite_score", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo11_ai_automate_routine_tasks_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo12_ai_optimize_workflows_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo13_ai_predict_patient_demand_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo14_ai_predict_staff_needs_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "iv2_physical_environment_score", "moderator": "mo15_ai_staff_scheduling_pct", "outcome": "dv15_preventable_stays_rate"},
        # --- Newly requested tests (added) ---
        {"iv": "mo1_genai_composite_score", "moderator": "mo2_robotics_composite_score", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo11_ai_automate_routine_tasks_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo12_ai_optimize_workflows_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo13_ai_predict_patient_demand_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo14_ai_predict_staff_needs_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
        {"iv": "mo15_ai_staff_scheduling_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv15_preventable_stays_rate"},
    ]
    
    interaction_results = []
    for spec in interaction_specs:
        iv_col, mo_col, y_col = spec["iv"], spec["moderator"], spec["outcome"]
        iv_col_c, mo_col_c = f"{iv_col}_c", f"{mo_col}_c"
        logger.info(f"  Interaction test: {iv_col} x {mo_col} -> {y_col}")

        required_cols = [iv_col_c, mo_col_c, y_col, 'state_fips_for_clustering'] + base_controls
        if any(c not in df.columns for c in required_cols):
            logger.warning(f"    Skipping: missing one or more required columns.")
            continue
            
        d = df[required_cols].dropna()
        if d.empty:
            logger.warning("    Skipping: Not enough data after dropping NaNs.")
            continue
            
        interaction_term = f"{iv_col_c}_x_{mo_col_c}"
        
        # Ensure original IVs are not duplicated in controls
        controls_for_model = [c for c in base_controls if c not in [iv_col, mo_col]]
        
        X = d[[iv_col_c, mo_col_c] + controls_for_model].copy()
        X[interaction_term] = d[iv_col_c] * d[mo_col_c]
        
        res = run_ols_clustered(d[y_col], X, d['state_fips_for_clustering'])
        beta = res.params.get(interaction_term, np.nan)
        pval = res.pvalues.get(interaction_term, np.nan)
        ci = res.conf_int().loc[interaction_term].values if interaction_term in res.params.index else (np.nan, np.nan)

        interaction_results.append({
            "IV": iv_col, "Moderator": mo_col, "Outcome": y_col, "N": len(d),
            "Interaction_Beta": beta, "Interaction_p": pval,
            "Interaction_CI_Lower": ci[0], "Interaction_CI_Upper": ci[1],
        })
        
        plot_interaction_continuous(d, iv_col_c, mo_col_c, y_col, 
                                    name=f"{spec['iv'].split('_')[0]}_x_{spec['moderator'].split('_')[0]}", 
                                    beta=beta, pval=pval, plot_dir=plot_dir, logger=logger)

    # --- NEW: Supplementary Interaction tests for DV21 (Premature Death) ---
    interaction_specs_dv21 = [
        {"iv": "mo1_genai_composite_score", "moderator": "mo2_robotics_composite_score", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo11_ai_automate_routine_tasks_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo12_ai_optimize_workflows_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo13_ai_predict_patient_demand_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo14_ai_predict_staff_needs_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
        {"iv": "mo15_ai_staff_scheduling_pct", "moderator": "mo21_robotics_in_hospital_pct", "outcome": "dv21_premature_death_ypll_rate"},
    ]

    # Process regular interaction tests
    for spec in interaction_specs:
        iv_col, mo_col, y_col = spec["iv"], spec["moderator"], spec["outcome"]
        iv_col_c, mo_col_c = f"{iv_col}_c", f"{mo_col}_c"
        logger.info(f"  Interaction test: {iv_col} x {mo_col} -> {y_col}")

        required_cols = [iv_col_c, mo_col_c, y_col, 'state_fips_for_clustering'] + base_controls
        if any(c not in df.columns for c in required_cols):
            logger.warning(f"    Skipping: missing one or more required columns.")
            continue
            
        d = df[required_cols].dropna()
        if d.empty:
            logger.warning("    Skipping: Not enough data after dropping NaNs.")
            continue
            
        interaction_term = f"{iv_col_c}_x_{mo_col_c}"
        
        # Ensure original IVs are not duplicated in controls
        controls_for_model = [c for c in base_controls if c not in [iv_col, mo_col]]
        
        X = d[[iv_col_c, mo_col_c] + controls_for_model].copy()
        X[interaction_term] = d[iv_col_c] * d[mo_col_c]
        
        res = run_ols_clustered(d[y_col], X, d['state_fips_for_clustering'])
        beta = res.params.get(interaction_term, np.nan)
        pval = res.pvalues.get(interaction_term, np.nan)
        ci = res.conf_int().loc[interaction_term].values if interaction_term in res.params.index else (np.nan, np.nan)

        interaction_results.append({
            "IV": iv_col, "Moderator": mo_col, "Outcome": y_col, "N": len(d),
            "Interaction_Beta": beta, "Interaction_p": pval,
            "Interaction_CI_Lower": ci[0], "Interaction_CI_Upper": ci[1],
        })
        
        plot_interaction_continuous(d, iv_col_c, mo_col_c, y_col, 
                                   name=f"{spec['iv'].split('_')[0]}_x_{spec['moderator'].split('_')[0]}", 
                                   beta=beta, pval=pval, plot_dir=plot_dir, logger=logger)

    # Process DV21 interaction tests
    logger.info("  Running DV21 (premature death) interaction tests...")
    for spec in interaction_specs_dv21:
        iv_col, mo_col, y_col = spec["iv"], spec["moderator"], spec["outcome"]
        iv_col_c, mo_col_c = f"{iv_col}_c", f"{mo_col}_c"
        logger.info(f"  DV21 Interaction test: {iv_col} x {mo_col} -> {y_col}")

        required_cols = [iv_col_c, mo_col_c, y_col, 'state_fips_for_clustering'] + base_controls
        if any(c not in df.columns for c in required_cols):
            logger.warning(f"    Skipping: missing one or more required columns.")
            continue
            
        d = df[required_cols].dropna()
        if d.empty:
            logger.warning("    Skipping: Not enough data after dropping NaNs.")
            continue
            
        interaction_term = f"{iv_col_c}_x_{mo_col_c}"
        
        # Ensure original IVs are not duplicated in controls
        controls_for_model = [c for c in base_controls if c not in [iv_col, mo_col]]
        
        X = d[[iv_col_c, mo_col_c] + controls_for_model].copy()
        X[interaction_term] = d[iv_col_c] * d[mo_col_c]
        
        res = run_ols_clustered(d[y_col], X, d['state_fips_for_clustering'])
        beta = res.params.get(interaction_term, np.nan)
        pval = res.pvalues.get(interaction_term, np.nan)
        ci = res.conf_int().loc[interaction_term].values if interaction_term in res.params.index else (np.nan, np.nan)

        interaction_results.append({
            "IV": iv_col, "Moderator": mo_col, "Outcome": y_col, "N": len(d),
            "Interaction_Beta": beta, "Interaction_p": pval,
            "Interaction_CI_Lower": ci[0], "Interaction_CI_Upper": ci[1],
        })
        
        plot_interaction_continuous(d, iv_col_c, mo_col_c, y_col, 
                                   name=f"{spec['iv'].split('_')[0]}_x_{spec['moderator'].split('_')[0]}_on_DV21", 
                                   beta=beta, pval=pval, plot_dir=plot_dir, logger=logger)

    dissertation_interaction_df = pd.DataFrame(interaction_results)
    if not dissertation_interaction_df.empty:
        out_csv = os.path.join(out_dir, "dissertation_interaction_effects_summary.csv")
        dissertation_interaction_df.to_csv(out_csv, index=False)
        logger.info(f"Saved dissertation interaction effects summary to {out_csv}")
    
    # --------------------------------------------------------------------------------
    # PART I. Assemble Unified Report
    # --------------------------------------------------------------------------------
    logger.info("PART I: Assembling final unified report.")
    assemble_replication_report_full(
        out_dir=out_dir,
        idx_df=idx_df,
        h1h4_df=h1h4_df,
        capex_df=capex_df,
        dissertation_direct_df=dissertation_direct_df,
        dissertation_interaction_df=dissertation_interaction_df,
        logger=logger
    )

    logger.info("All done. Check logs and output directory for results.")

if __name__ == "__main__":
    main()