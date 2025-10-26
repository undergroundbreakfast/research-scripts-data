#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
# -*- coding: utf-8 -*-
"""
Negative Controls & Placebos for Dissertation (NC suite v2 with MO1 components)
-------------------------------------------------------------------------------
Drop-in, end-to-end script to run negative control outcome (NCO) tests,
negative control exposure (NCE) interactions, placebo outcomes for your
IV3×MO moderation, temporal falsification (pre-exposure leads), and
permutation/randomization inference—mirroring your pipeline:

- Controls: log(pop), MedicaidExpansion, IV2, IV4, and Census Division FE
- Clustered SEs by state
- Centering of continuous variables used in interactions
- BH-FDR within families; pass if p>.10 and q>.25
- NEW: Everywhere MO1 (composite) was tested, we also run MO11–MO15 components

Outputs CSVs and diagnostic plots in ./negative_controls_output

Author: Aaron Johnson (NC suite assembled) + component expansion
"""

import os
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import traceback
from typing import Dict, List, Tuple, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection

# Optional viz
try:
    import matplotlib.pyplot as plt
    VISUALS = True
except Exception:
    VISUALS = False

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(prefix="negative_controls_log"):
    logger = logging.getLogger("nc_tests")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(os.path.join("logs", f"{prefix}_{ts}.txt"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(fh); logger.addHandler(ch)
    logger.info("Logger initialized.")
    return logger

def log_sql(sql_text, params=None, logger=None):
    """Log SQL statement with optional parameters."""
    if logger is None:
        return
    
    # Format the SQL query nicely
    formatted_sql = str(sql_text).strip()
    
    # Log the raw SQL
    logger.info("=" * 80)
    logger.info("EXECUTING SQL:")
    logger.info("-" * 80)
    logger.info(formatted_sql)
    
    # Log parameters if provided
    if params:
        logger.info("-" * 80)
        logger.info(f"PARAMETERS: {params}")
    
    logger.info("=" * 80)

# -----------------------------------------------------------------------------
# DB connection helpers
# -----------------------------------------------------------------------------
def connect_to_database(logger) -> Engine:
    host = os.getenv("POSTGRES_HOST", "localhost")
    database = os.getenv("POSTGRES_DB", "Research_TEST")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRESQL_KEY")
    if not password:
        logger.error("POSTGRESQL_KEY not set.")
        sys.exit("Database password not configured.")

    try:
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL '{database}'.")
        return engine
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

def table_exists(engine: Engine, name: str, logger=None) -> bool:
    sql = """
        SELECT 1 FROM information_schema.tables
        WHERE table_schema='public' AND table_name=:t LIMIT 1
    """
    log_sql(sql, {"t": name}, logger)
    try:
        with engine.connect() as conn:
            row = conn.execute(text(sql), {"t": name}).fetchone()
        return row is not None
    except Exception:
        return False

def list_columns(engine: Engine, name: str, logger=None) -> List[str]:
    sql = """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema='public' AND table_name=:t
    """
    log_sql(sql, {"t": name}, logger)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql), {"t": name}).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []

# -----------------------------------------------------------------------------
# Variable sets (NCOs & NCEs)
# -----------------------------------------------------------------------------
# Placebo outcomes (DVs that should not move)
PLACEBO_DVS = [
    "voter_turnout_raw_value",
    "census_participation_raw_value",
    "traffic_volume_raw_value",
    "gender_pay_gap_raw_value",
    "reading_scores_raw_value",
    "math_scores_raw_value",
    "long_commute_driving_alone_raw_value",
]

# Direct-effect NCOs (expanded, includes structural/env/demographic)
DIRECT_NCO_DVS = sorted(set(PLACEBO_DVS + [
    "driving_alone_to_work_raw_value",
    "school_funding_adequacy_raw_value",
    "school_segregation_raw_value",
    "homeownership_raw_value",
    "severe_housing_cost_burden_raw_value",
    # FIX: Use the exact column names from database
    # These are in chunk_2, not chunk_3
    "air_pollution_particulate_matter_raw_value",
    "drinking_water_violations_raw_value",
    "not_proficient_in_english_raw_value",
    "female_raw_value",
    "below_18_years_of_age_raw_value",
    "_65_and_older_raw_value",
    "non_hispanic_black_raw_value",
    "hispanic_raw_value",
    "non_hispanic_white_raw_value",
    "juvenile_arrests_raw_value"
]))

# Placebo IVs to replace IV3 in moderation (explicitly exclude broadband)
PLACEBO_IVS = [
    "voter_turnout_raw_value",
    "census_participation_raw_value",
    "traffic_volume_raw_value",
    "driving_alone_to_work_raw_value",
    "long_commute_driving_alone_raw_value",
    "reading_scores_raw_value",
    "math_scores_raw_value",
    "school_funding_adequacy_raw_value",
    "school_segregation_raw_value",
    "gender_pay_gap_raw_value",
]

# Outcomes of interest (core)
CORE_DV21 = "dv21_premature_death_ypll_rate"       # from conceptual variables or chunk_1
CORE_DV2  = "dv2_health_outcomes_score"            # from conceptual model (index)
LIFE_EXP  = "life_expectancy_raw_value"            # for guardrail (from chunk_2)
IV3       = "iv3_health_behaviors_score"           # main IV for moderation
MO1       = "mo1_genai_composite_score"
MO2       = "mo2_robotics_composite_score"

# --- NEW: MO1 component moderators created in fetch_base via renames ---
MO11 = "mo11_ai_automate_routine_tasks_pct"
MO12 = "mo12_ai_optimize_workflows_pct"
MO13 = "mo13_ai_predict_patient_demand_pct"
MO14 = "mo14_ai_predict_staff_needs_pct"
MO15 = "mo15_ai_staff_scheduling_pct"

# All moderators to evaluate where MO1 was previously tested
MO1_COMPONENTS = [MO11, MO12, MO13, MO14, MO15]
MO1_FAMILY = [MO1] + MO1_COMPONENTS

# Short codes for clean output labels
MO_CODE = {
    MO1:  "MO1",
    MO11: "MO11",
    MO12: "MO12",
    MO13: "MO13",
    MO14: "MO14",
    MO15: "MO15",
    MO2:  "MO2"
}

# -----------------------------------------------------------------------------
# Data fetch
# -----------------------------------------------------------------------------
def fetch_base(engine: Engine, logger) -> pd.DataFrame:
    """
    Fetches base (vcm, vcv optional), tech (vcts), HRSA rurality, AHA CAPEX.
    Does NOT fetch CHR analytic chunk X yet; those are fetched separately.
    """
    # choose tables/views if they exist
    vcm = "vw_conceptual_model_adjpd" if table_exists(engine, "vw_conceptual_model_adjpd") \
        else ("vw_conceptual_model" if table_exists(engine, "vw_conceptual_model") else None)
    vcv = "vw_conceptual_model_variables_adjpd" if table_exists(engine, "vw_conceptual_model_variables_adjpd") \
        else ("vw_conceptual_model_variables" if table_exists(engine, "vw_conceptual_model_variables") else None)
    vcts = "vw_county_tech_summary_adjpd" if table_exists(engine, "vw_county_tech_summary_adjpd") \
        else ("vw_county_tech_summary" if table_exists(engine, "vw_county_tech_summary") else None)

    if vcm is None:
        logger.error("Missing conceptual model view (vcm).")
        sys.exit(1)

    tech_cols = list_columns(engine, vcts) if vcts else []
    # resolve tech columns (composite present on vcm; include components if available)
    alias_map = {
        "pct_wfaiart_enabled": ["pct_wfaiart_enabled_adjpd", "pct_wfaiart_enabled"],
        "pct_wfaioacw_enabled": ["pct_wfaioacw_enabled_adjpd", "pct_wfaioacw_enabled"],
        "pct_wfaippd_enabled": ["pct_wfaippd_enabled_adjpd", "pct_wfaippd_enabled"],
        "pct_wfaipsn_enabled": ["pct_wfaipsn_enabled_adjpd", "pct_wfaipsn_enabled"],
        "pct_wfaiss_enabled": ["pct_wfaiss_enabled_adjpd", "pct_wfaiss_enabled"],
        "pct_robohos_enabled": ["pct_robohos_enabled_adjpd", "pct_robohos_enabled"],
    }
    def resolve(a):
        for c in alias_map[a]:
            if c in tech_cols:
                return c
        return None

    tech_selects = []
    # NOTE: Keep your existing, consistent mapping of fields -> named columns
    renames = {
        "pct_wfaiart_enabled":"mo11_ai_automate_routine_tasks_pct",
        "pct_wfaioacw_enabled":"mo12_ai_optimize_workflows_pct",
        "pct_wfaippd_enabled":"mo13_ai_predict_patient_demand_pct",
        "pct_wfaipsn_enabled":"mo14_ai_predict_staff_needs_pct",
        "pct_wfaiss_enabled":"mo15_ai_staff_scheduling_pct",
        "pct_robohos_enabled":"mo21_robotics_in_hospital_pct"
    }
    for key, outname in renames.items():
        if vcts:
            col = resolve(key)
            if col:
                tech_selects.append(f"vcts.{col}::numeric AS {outname}")
            else:
                tech_selects.append(f"NULL::numeric AS {outname}")

    hrsa = "hrsa_health_equity_data" if table_exists(engine, "hrsa_health_equity_data") else None
    aha  = "aha_survey_data" if table_exists(engine, "aha_survey_data") else None

    sql = [f"""
        SELECT
          vcm.county_fips,
          vcm.population::numeric AS population,
          vcm.census_division,
          vcm.medicaid_expansion_active AS iv1_medicaid_expansion_active,
          vcm.physical_environment_score::numeric AS iv2_physical_environment_score,
          vcm.health_behaviors_score::numeric AS iv3_health_behaviors_score,
          vcm.social_economic_factors_score::numeric AS iv4_social_economic_factors_score,
          vcm.health_outcomes_score::numeric AS dv2_health_outcomes_score,
          vcm.weighted_ai_adoption_score::numeric AS mo1_genai_composite_score,
          vcm.weighted_robotics_adoption_score::numeric AS mo2_robotics_composite_score
    """]
    if vcv and "premature_death_raw_value" in list_columns(engine, vcv):
        sql.append(f", vcv.premature_death_raw_value::numeric AS {CORE_DV21}")
    else:
        sql.append(f", NULL::numeric AS {CORE_DV21}")
    if vcts:
        sql.append(", " + ", ".join(tech_selects))
    if hrsa:
        sql.append(", hrsa.irr_county_value::numeric AS sp5_irr_county_value")
    else:
        sql.append(", NULL::numeric AS sp5_irr_county_value")
    if aha:
        sql.append(""",
          aha.capex_sum::numeric AS fi1_capex_sum,
          aha.adjpd_sum::numeric AS fi2_adjpd_sum,
          CASE WHEN aha.adjpd_sum IS NOT NULL AND aha.adjpd_sum<>0
               THEN (aha.capex_sum::numeric/aha.adjpd_sum::numeric)
               ELSE NULL END AS fi_capex_intensity_ratio
        """)
    else:
        sql.append(", NULL::numeric AS fi1_capex_sum, NULL::numeric AS fi2_adjpd_sum, NULL::numeric AS fi_capex_intensity_ratio")

    sql.append(f"FROM public.{vcm} vcm")
    if vcv:  sql.append(f"LEFT JOIN public.{vcv}  vcv  ON vcm.county_fips=vcv.county_fips")
    if vcts: sql.append(f"LEFT JOIN public.{vcts} vcts ON vcm.county_fips=vcts.county_fips")
    if hrsa:
        sql.append(f"""
        LEFT JOIN (
          SELECT LPAD(TRIM(CAST(county_fips_code AS TEXT)),5,'0') AS county_fips,
                 AVG(NULLIF(irr_county_value,'')::numeric) AS irr_county_value
          FROM public.{hrsa}
          GROUP BY 1
        ) hrsa ON vcm.county_fips=hrsa.county_fips
        """)
    if aha:
        sql.append(f"""
        LEFT JOIN (
          SELECT LPAD(TRIM(CAST(fcounty AS TEXT)),5,'0') AS county_fips,
                 SUM(NULLIF(ceamt,'')::numeric) AS capex_sum,
                 SUM(NULLIF(adjpd,'')::numeric) AS adjpd_sum
          FROM public.{aha}
          GROUP BY 1
        ) aha ON vcm.county_fips=aha.county_fips
        """)
    sql.append("WHERE vcm.population IS NOT NULL AND vcm.population::numeric>0;")

    final_sql = "\n".join(sql)
    log_sql(final_sql, None, logger)
    
    df = pd.read_sql_query(text(final_sql), engine)
    if df.empty:
        logger.error("Base fetch returned 0 rows.")
        sys.exit(1)

    # hygiene
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df = df[~df["county_fips"].str.endswith("000")]  # drop state-level aggregates
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df[df["population"].notna() & (df["population"] > 0)]
    logger.info(f"Base data: {df.shape[0]} counties.")

    return df

def fetch_chr_chunk(engine: Engine, table: str, cols: List[str], logger) -> pd.DataFrame:
    """
    Fetch county_fips, year, and requested columns from chr_analytic_chunk_{1,2,3}.
    Coerces to numeric where possible, handling empty strings and invalid values.
    """
    if not table_exists(engine, table):
        logger.warning(f"{table} not found; returning empty.")
        return pd.DataFrame(columns=["county_fips","year"]+cols)

    # Use _5_digit_fips as the key, but keep 'county_fips' in the list for compatibility
    keep = set(["county_fips", "year"] + cols)
    existing = list_columns(engine, table)
    cols_sel = [c for c in cols if c in existing]

    # If none present, just return (no-op)
    if not cols_sel:
        logger.warning(f"{table}: none of requested columns present; returning empty.")
        return pd.DataFrame(columns=["county_fips","year"]+cols)

    # Build SQL with explicit NULLIF to convert empty strings to NULL
    # This is more reliable than doing it in pandas
    cleaned_cols = []
    for c in cols_sel:
        # Use NULLIF to convert empty strings to NULL, then try to cast to numeric
        # If cast fails, it will return NULL
        cleaned_cols.append(f"NULLIF(TRIM({c}), '') AS {c}")
    
    sel = ", ".join(cleaned_cols)
    
    # CRITICAL FIX: Use _5_digit_fips as the key and alias it to county_fips
    sql_query = f"""
    SELECT 
        LPAD(TRIM(CAST(_5_digit_fips AS TEXT)), 5, '0') as county_fips,
        CAST(year AS TEXT) as year,
        {sel}
    FROM public.{table}
    WHERE year IS NOT NULL
    """
    
    log_sql(sql_query, None, logger)
    
    try:
        df = pd.read_sql_query(text(sql_query), engine)
    except Exception as e:
        logger.error(f"Error fetching from {table}: {e}")
        logger.warning(f"Falling back to basic query without NULLIF cleaning")
        # Fallback to basic query
        basic_sel = ", ".join(cols_sel)
        sql_query = f"""
        SELECT 
            LPAD(TRIM(CAST(_5_digit_fips AS TEXT)), 5, '0') as county_fips,
            CAST(year AS TEXT) as year,
            {basic_sel}
        FROM public.{table}
        WHERE year IS NOT NULL
        """
        log_sql(sql_query, None, logger)
        df = pd.read_sql_query(text(sql_query), engine)
    
    if df.empty:
        logger.warning(f"{table}: returned 0 rows.")
        return df
    
    # Additional safety: ensure county_fips is properly formatted in pandas too
    df["county_fips"] = df["county_fips"].astype(str).str.strip().str.zfill(5)
    
    # Filter out invalid county_fips (state-level aggregates, etc.)
    initial_count = len(df)
    df = df[~df["county_fips"].str.endswith("000")]  # Remove state-level
    df = df[df["county_fips"].str.match(r'^\d{5}$')]  # Keep only valid 5-digit codes
    filtered_count = len(df)
    
    if filtered_count < initial_count:
        logger.info(f"{table}: Filtered {initial_count - filtered_count} invalid county_fips records")
    
    # Aggressive numeric coercion for data columns
    for c in cols_sel:
        if c in df.columns:
            # Step 1: Convert to string and clean
            series = df[c].astype(str)
            
            # Step 2: Replace common non-numeric markers with NaN
            series = series.replace({
                'None': np.nan,
                'nan': np.nan,
                'NaN': np.nan,
                'NULL': np.nan,
                'null': np.nan,
                '': np.nan,
                ' ': np.nan,
                'NA': np.nan,
                'N/A': np.nan,
                '#DIV/0!': np.nan,
                '#VALUE!': np.nan,
                '#REF!': np.nan,
                '#NUM!': np.nan,
                '-': np.nan,
                '--': np.nan,
                '---': np.nan,
            })
            
            # Step 3: Strip whitespace
            series = series.str.strip()
            
            # Step 4: Replace empty strings with NaN
            series = series.replace('', np.nan)
            
            # Step 5: Convert to numeric, coercing errors to NaN
            df[c] = pd.to_numeric(series, errors="coerce")
            
            # Log conversion stats
            non_null = df[c].notna().sum()
            total = len(df)
            if non_null == 0:
                logger.debug(f"{table}.{c}: No valid numeric values found (0/{total})")
            elif non_null < total:
                logger.debug(f"{table}.{c}: {non_null}/{total} valid numeric values")
    
    # Convert year to numeric, handling various formats
    # First clean the year string (remove any non-numeric characters)
    df["year_clean"] = df["year"].astype(str).str.extract(r'(\d{4})', expand=False)
    df["year_num"] = pd.to_numeric(df["year_clean"], errors="coerce")
    
    # Log year range found
    if df["year_num"].notna().any():
        min_year = df["year_num"].min()
        max_year = df["year_num"].max()
        unique_years = sorted(df["year_num"].dropna().unique())
        logger.info(f"{table}: Years range from {int(min_year)} to {int(max_year)} " +
                   f"({df['year_num'].notna().sum()} valid records, years: {[int(y) for y in unique_years]})")
    else:
        logger.warning(f"{table}: No valid numeric years found")
    
    # Log which columns have data
    cols_with_data = [c for c in cols_sel if df[c].notna().sum() > 0]
    if cols_with_data:
        logger.info(f"{table}: {len(cols_with_data)}/{len(cols_sel)} requested columns have data: {cols_with_data[:5]}{'...' if len(cols_with_data) > 5 else ''}")
    else:
        logger.warning(f"{table}: None of the requested columns have valid data!")
    
    # DIAGNOSTIC: Log sample county_fips after processing
    logger.info(f"{table}: Sample county_fips after processing: {df['county_fips'].head().tolist()}")
    
    return df

def latest_by_county(df: pd.DataFrame, key="county_fips") -> pd.DataFrame:
    if df.empty or "year_num" not in df.columns:
        return df
    # keep latest numeric year per county
    # CRITICAL FIX: Use sort_index after setting index to avoid reordering issues
    # that can be caused by groupby on string-numeric keys.
    d = df.dropna(subset=["year_num"]).sort_values([key, "year_num"])
    d = d.groupby(key, as_index=False, sort=False).tail(1)
    d = d.drop(columns=["year_num"])
    return d

# -----------------------------------------------------------------------------
# Preparation, FE, clustering, helpers
# -----------------------------------------------------------------------------
def add_const(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant='add')

def run_ols_clustered(Y: pd.Series, X: pd.DataFrame, clusters: pd.Series):
    model = sm.OLS(Y, add_const(X))
    try:
        if clusters.nunique() >= 2:
            return model.fit(cov_type="cluster", cov_kwds={"groups": clusters})
        else:
            return model.fit()
    except Exception:
        return model.fit()

def bh_fdr(series_p: pd.Series) -> pd.Series:
    mask = series_p.notna()
    q = pd.Series(np.nan, index=series_p.index)
    if mask.any():
        _, qvals = fdrcorrection(series_p[mask].values, alpha=0.05, method='indep')
        q.loc[mask] = qvals
    return q

def winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    if s.dropna().size < 25:  # avoid degenerate quantiles in tiny samples
        return s
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=ql, upper=qh)

def center(s: pd.Series) -> pd.Series:
    return s - s.mean()

def ensure_centered(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns and f"{c}_c" not in df.columns:
            df[f"{c}_c"] = center(df[c])
    return df

def make_division_dummies(df: pd.DataFrame, logger) -> Tuple[pd.DataFrame, List[str]]:
    dummies = []
    if "census_division" in df.columns:
        df["census_division"] = df["census_division"].astype(str)
        D = pd.get_dummies(df["census_division"], prefix="div", drop_first=True, dtype=int)
        df = pd.concat([df, D], axis=1)
        dummies = list(D.columns)
        logger.info(f"Added division FE (dummies={len(dummies)}).")
    else:
        logger.warning("census_division missing; FE omitted.")
    return df, dummies

def prep_common(df: pd.DataFrame, logger) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    # cluster grouping
    if "county_fips" not in df.columns:
        logger.error("county_fips missing.")
        sys.exit(1)
    df["state_fips_for_clustering"] = df["county_fips"].astype(str).str[:2]

    # log(pop)
    if "population" not in df.columns:
        logger.error("population missing.")
        sys.exit(1)
    df = df[df["population"] > 0].copy()
    df["log_population"] = np.log(df["population"])

    # Medicaid expansion to {0,1}
    med = "iv1_medicaid_expansion_active"
    if med in df.columns:
        s = df[med].astype(str).str.lower().map(
            {"true":1,"t":1,"yes":1,"y":1,"1":1,"false":0,"f":0,"no":0,"n":0,"0":0}
        )
        df[med] = pd.to_numeric(s, errors="coerce")
    else:
        df[med] = np.nan

    # FE
    df, div_fe = make_division_dummies(df, logger)

    base_controls = [
        "iv4_social_economic_factors_score",
        "iv2_physical_environment_score",
        "iv3_health_behaviors_score",
        "iv1_medicaid_expansion_active",
        "log_population",
        *div_fe
    ]
    base_controls = [c for c in base_controls if c in df.columns]
    return df, base_controls

# -----------------------------------------------------------------------------
# Test suites (expanded to MO1 components)
# -----------------------------------------------------------------------------
def run_direct_nco_tests(df: pd.DataFrame,
                         nco_latest_c3: pd.DataFrame,
                         nco_latest_c2: pd.DataFrame,
                         base_controls: List[str],
                         out_dir: str,
                         logger) -> pd.DataFrame:

    # Ensure county_fips is lowercase in CHR chunks to match base_df
    if not nco_latest_c3.empty:
        nco_latest_c3 = nco_latest_c3.copy()
        nco_latest_c3.columns = [c.lower() for c in nco_latest_c3.columns]
        # DIAGNOSTIC: Check county_fips format
        logger.info(f"CHR chunk_3: {len(nco_latest_c3)} counties")
        logger.info(f"CHR chunk_3 sample county_fips: {nco_latest_c3['county_fips'].head().tolist()}")
        logger.info(f"CHR chunk_3 county_fips dtype: {nco_latest_c3['county_fips'].dtype}")
        
    if not nco_latest_c2.empty:
        nco_latest_c2 = nco_latest_c2.copy()
        nco_latest_c2.columns = [c.lower() for c in nco_latest_c2.columns]
        logger.info(f"CHR chunk_2: {len(nco_latest_c2)} counties")
        logger.info(f"CHR chunk_2 sample county_fips: {nco_latest_c2['county_fips'].head().tolist()}")
        logger.info(f"CHR chunk_2 county_fips dtype: {nco_latest_c2['county_fips'].dtype}")
    
    # DIAGNOSTIC: Check base df county_fips
    logger.info(f"Base df: {len(df)} counties")
    logger.info(f"Base df sample county_fips: {df['county_fips'].head().tolist()}")
    logger.info(f"Base df county_fips dtype: {df['county_fips'].dtype}")
    
    # DIAGNOSTIC: Check for overlapping county_fips
    if not nco_latest_c3.empty:
        base_fips = set(df['county_fips'].unique())
        chr_fips = set(nco_latest_c3['county_fips'].unique())
        overlap = base_fips & chr_fips
        logger.info(f"County overlap between base and chunk_3: {len(overlap)} counties")
        if len(overlap) < 10:
            logger.warning(f"Very few overlapping counties! Base has {len(base_fips)}, CHR has {len(chr_fips)}")
            logger.warning(f"Sample base county_fips: {list(base_fips)[:5]}")
            logger.warning(f"Sample CHR county_fips: {list(chr_fips)[:5]}")

    div_cols = [c for c in df.columns if c.startswith('div_')]

    keep = ["county_fips","state_fips_for_clustering","census_division",
            "iv1_medicaid_expansion_active","iv2_physical_environment_score",
            "iv3_health_behaviors_score","iv4_social_economic_factors_score",
            "log_population", MO2] + MO1_FAMILY + div_cols

    d = df[[c for c in keep if c in df.columns]].copy()
    d = ensure_centered(d, MO1_FAMILY + [MO2])
    
    # CRITICAL FIX: Don't set index yet - do a merge instead
    # The issue is that after setting index, the join isn't matching properly
    
    rows = []

    for dv in DIRECT_NCO_DVS:
        # Instead of join, use merge to be more explicit
        if not nco_latest_c3.empty and dv in nco_latest_c3.columns:
            # Merge instead of join
            dloc = d.merge(nco_latest_c3[['county_fips', dv]], on='county_fips', how='left')
        elif not nco_latest_c2.empty and dv in nco_latest_c2.columns:
            dloc = d.merge(nco_latest_c2[['county_fips', dv]], on='county_fips', how='left')
        else:
            logger.info(f"[Direct NCO] Skipping {dv}: not found in any chunk.")
            continue
        
        # Check how many non-null values we have after merge
        n_valid = dloc[dv].notna().sum()
        n_total = len(dloc)
        
        # DIAGNOSTIC: More detailed logging
        if n_valid == 0:
            logger.warning(f"[Direct NCO] Skipping {dv}: 0 valid values after merge.")
            logger.warning(f"  Total rows after merge: {n_total}")
            logger.warning(f"  DV column type: {dloc[dv].dtype}")
            logger.warning(f"  Sample DV values: {dloc[dv].head().tolist()}")
            continue
        else:
            logger.info(f"[Direct NCO] Processing {dv}: {n_valid}/{n_total} valid values ({100*n_valid/n_total:.1f}%).")

        y_w = winsorize(dloc[dv])

        # (a) Composite MO1 and MO2 (original behavior, kept)
        cols = [dv, f"{MO1}_c", f"{MO2}_c", "state_fips_for_clustering"] + base_controls
        data = dloc[[c for c in cols if c in dloc.columns]].dropna()
        
        logger.info(f"[Direct NCO] {dv} with MO1+MO2: {len(data)} counties after dropna")
        
        if not data.empty:
            X = data[[f"{MO1}_c", f"{MO2}_c"] + [c for c in base_controls if c in data.columns]]
            res = run_ols_clustered(y_w.loc[data.index], X, data["state_fips_for_clustering"])
            for t in [f"{MO1}_c", f"{MO2}_c"]:
                if t in res.params.index:
                    ci = res.conf_int().loc[t].values
                else:
                    ci = (np.nan, np.nan)
                rows.append({
                    "Family":"Direct NCO",
                    "Outcome": dv,
                    "Treatment": t,
                    "Moderator": MO_CODE[MO1] if t.startswith(MO1) else MO_CODE[MO2],
                    "N": len(data),
                    "Beta": res.params.get(t, np.nan),
                    "p": res.pvalues.get(t, np.nan),
                    "CI_lower": ci[0], "CI_upper": ci[1]
                })

        # (b) Components MO11–MO15 paired with MO2 in the same model
        for m in MO1_COMPONENTS:
            if m not in dloc.columns:
                logger.debug(f"[Direct NCO] {m} missing in dloc; skipping.")
                continue
            cols = [dv, f"{m}_c", f"{MO2}_c", "state_fips_for_clustering"] + base_controls
            data = dloc[[c for c in cols if c in dloc.columns]].dropna()
            if data.empty:
                continue
            X = data[[f"{m}_c", f"{MO2}_c"] + [c for c in base_controls if c in data.columns]]
            res = run_ols_clustered(y_w.loc[data.index], X, data["state_fips_for_clustering"])
            t = f"{m}_c"
            if t in res.params.index:
                ci = res.conf_int().loc[t].values
            else:
                ci = (np.nan, np.nan)
            rows.append({
                "Family":"Direct NCO",
                "Outcome": dv,
                "Treatment": t,
                "Moderator": MO_CODE[m],
                "N": len(data),
                "Beta": res.params.get(t, np.nan),
                "p": res.pvalues.get(t, np.nan),
                "CI_lower": ci[0], "CI_upper": ci[1]
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = bh_fdr(out["p"])
        out["Pass_Conservative"] = (out["p"] > 0.10) & (out["q"] > 0.25)
        out.sort_values(["Outcome","Moderator","Treatment"]).to_csv(
            os.path.join(out_dir, "direct_nco_results.csv"), index=False)
        logger.info(f"Wrote direct NCO results ({len(out)} rows).")
    else:
        logger.warning("Direct NCO tests produced 0 results - check data availability and merge logic.")
    return out

def run_placebo_interactions_replace_iv3(df: pd.DataFrame,
                                         latest_c3: pd.DataFrame,
                                         latest_c2: pd.DataFrame,
                                         base_controls: List[str],
                                         out_dir: str,
                                         logger) -> pd.DataFrame:
    # Lowercase column names
    if not latest_c3.empty:
        latest_c3 = latest_c3.copy()
        latest_c3.columns = [c.lower() for c in latest_c3.columns]
    if not latest_c2.empty:
        latest_c2 = latest_c2.copy()
        latest_c2.columns = [c.lower() for c in latest_c2.columns]
    
    div_cols = [c for c in df.columns if c.startswith('div_')]

    cols_to_keep = ["county_fips","state_fips_for_clustering","census_division",
                    "iv1_medicaid_expansion_active","iv2_physical_environment_score",
                    "iv3_health_behaviors_score","iv4_social_economic_factors_score",
                    "log_population", CORE_DV2, CORE_DV21, MO2] + MO1_FAMILY + div_cols

    d = df[[c for c in cols_to_keep if c in df.columns]].copy()

    rows = []

    for piv in PLACEBO_IVS:
        # add placebo IV - lowercase the column name
        piv_lower = piv.lower()
        
        # FIX: Use merge instead of join
        if not latest_c3.empty and piv_lower in latest_c3.columns:
            tmp = d.merge(latest_c3[['county_fips', piv_lower]], on='county_fips', how='left')
            tmp = tmp.rename(columns={piv_lower: piv}) # Rename back for consistency
        elif not latest_c2.empty and piv_lower in latest_c2.columns:
            tmp = d.merge(latest_c2[['county_fips', piv_lower]], on='county_fips', how='left')
            tmp = tmp.rename(columns={piv_lower: piv})
        else:
            logger.info(f"[Placebo IV] Skipping {piv}: not found.")
            continue

        n_valid = tmp[piv].notna().sum()
        if n_valid == 0:
            logger.warning(f"[Placebo IV] Skipping {piv}: 0 valid values after merge.")
            continue
        logger.info(f"[Placebo IV] Processing {piv}: {n_valid} valid values.")

        tmp = tmp.copy()
        tmp[f"{piv}_c"] = center(tmp[piv])
        tmp = ensure_centered(tmp, MO1_FAMILY + [MO2])

        for ycol in [CORE_DV2, CORE_DV21]:
            if ycol not in tmp.columns:
                continue

            current_controls = [c for c in base_controls if c != 'iv3_health_behaviors_score']

            for m in (MO1_FAMILY + [MO2]):
                mo = f"{m}_c"
                if mo not in tmp.columns:
                    continue
                term = f"{piv}_c_x_{mo}"
                cols = [ycol, f"{piv}_c", mo, "state_fips_for_clustering"] + current_controls
                dat = tmp[[c for c in cols if c in tmp.columns]].dropna()
                if dat.empty: 
                    continue
                X = dat[[f"{piv}_c", mo] + [c for c in current_controls if c in dat.columns]].copy()
                res = run_ols_clustered(dat[ycol], X.assign(**{term: dat[f"{piv}_c"]*dat[mo]}), dat["state_fips_for_clustering"])
                if term in res.params.index:
                    ci = res.conf_int().loc[term].values
                else:
                    ci = (np.nan, np.nan)
                rows.append({
                    "Family":"Placebo Interaction (replace IV3)",
                    "Outcome": ycol, "PlaceboIV": piv,
                    "Moderator": MO_CODE[m],
                    "N": len(dat),
                    "Interaction_Beta": res.params.get(term, np.nan),
                    "p": res.pvalues.get(term, np.nan),
                    "CI_lower": ci[0], "CI_upper": ci[1]
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = out.groupby(["Outcome","Moderator"])["p"].transform(bh_fdr)
        out["Pass_Conservative"] = (out["p"] > 0.10) & (out["q"] > 0.25)
        out.to_csv(os.path.join(out_dir, "placebo_interactions_replace_iv3.csv"), index=False)
        logger.info(f"Wrote placebo interactions (replace IV3) ({len(out)} rows).")
    else:
        logger.warning("Placebo interaction tests produced 0 results - check data availability.")
    return out

def run_placebo_outcomes_keep_iv3(df: pd.DataFrame,
                                  latest_c3: pd.DataFrame,
                                  latest_c2: pd.DataFrame,
                                  base_controls: List[str],
                                  out_dir: str,
                                  logger) -> pd.DataFrame:
    # FIX: Lowercase columns to match merge logic
    if not latest_c3.empty:
        latest_c3 = latest_c3.copy()
        latest_c3.columns = [c.lower() for c in latest_c3.columns]
    if not latest_c2.empty:
        latest_c2 = latest_c2.copy()
        latest_c2.columns = [c.lower() for c in latest_c2.columns]

    div_cols = [c for c in df.columns if c.startswith('div_')]

    d = df[["county_fips","state_fips_for_clustering","census_division",
            "iv1_medicaid_expansion_active","iv2_physical_environment_score",
            "iv3_health_behaviors_score","iv4_social_economic_factors_score",
            "log_population", MO2] + MO1_FAMILY + div_cols].copy()

    d = ensure_centered(d, [IV3] + MO1_FAMILY + [MO2])

    rows = []
    for dv in PLACEBO_DVS:
        dv_lower = dv.lower()
        # FIX: Use merge instead of join
        if not latest_c3.empty and dv_lower in latest_c3.columns:
            tmp = d.merge(latest_c3[['county_fips', dv_lower]], on='county_fips', how='left')
            tmp = tmp.rename(columns={dv_lower: dv})
        elif not latest_c2.empty and dv_lower in latest_c2.columns:
            tmp = d.merge(latest_c2[['county_fips', dv_lower]], on='county_fips', how='left')
            tmp = tmp.rename(columns={dv_lower: dv})
        else:
            logger.info(f"[Placebo DV] Skipping {dv}: not found.")
            continue

        if tmp[dv].notna().sum() == 0:
            logger.warning(f"[Placebo DV] Skipping {dv}: 0 valid values after merge.")
            continue

        tmp = tmp.copy()
        tmp[dv] = winsorize(tmp[dv])

        # Test moderators: MO1 composite + components + MO2
        for m in (MO1_FAMILY + [MO2]):
            mo = f"{m}_c"
            term = f"{IV3}_c_x_{mo}"
            cols = [dv, f"{IV3}_c", mo, "state_fips_for_clustering"] + base_controls
            dat = tmp[[c for c in cols if c in tmp.columns]].dropna()
            if dat.empty:
                continue
            X = dat[[f"{IV3}_c", mo] + base_controls].copy()
            res = run_ols_clustered(dat[dv], X.assign(**{term: dat[f"{IV3}_c"]*dat[mo]}), dat["state_fips_for_clustering"])
            if term in res.params.index:
                ci = res.conf_int().loc[term].values
            else:
                ci = (np.nan, np.nan)
            rows.append({
                "Family":"Placebo Outcome (keep IV3×MO)",
                "Outcome": dv,
                "Moderator": MO_CODE[m],
                "N": len(dat),
                "Interaction_Beta": res.params.get(term, np.nan),
                "p": res.pvalues.get(term, np.nan),
                "CI_lower": ci[0], "CI_upper": ci[1]
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = out.groupby(["Outcome","Moderator"])["p"].transform(bh_fdr)
        out["Pass_Conservative"] = (out["p"] > 0.10) & (out["q"] > 0.25)
        out.to_csv(os.path.join(out_dir, "placebo_outcomes_keep_iv3mo.csv"), index=False)
        logger.info(f"Wrote placebo outcomes (keep IV3×MO) ({len(out)} rows).")
    return out

def run_temporal_falsification(engine: Engine,
                               base_df: pd.DataFrame,
                               dv21_long_c1: pd.DataFrame,
                               pop_long_c3: pd.DataFrame,
                               base_controls: List[str],
                               out_dir: str,
                               logger) -> pd.DataFrame:
    """
    DV21_y (pre-2023) ~ MOx_2023_c + MO2_2023_c + log(pop)_y + IV2 + IV4 + Medicaid + FE
    where MOx ∈ {MO1 composite, MO11..MO15}
    
    NOTE: This test requires historical CHR data (2010-2022). If only 2024 data exists,
    this test will be skipped.
    """
    if dv21_long_c1.empty:
        logger.warning("Temporal falsification skipped: chunk_1 empty.")
        return pd.DataFrame()

    # Log what we have
    logger.info(f"DV21 long data shape: {dv21_long_c1.shape}")
    unique_years = sorted(dv21_long_c1['year_num'].dropna().unique())
    logger.info(f"DV21 unique years available: {[int(y) for y in unique_years]}")
    
    # Filter to reasonable year range and pre-2023
    cand_years = [int(y) for y in unique_years if 2010 <= int(y) <= 2022]
    
    if not cand_years:
        logger.warning(f"No years in range 2010-2022 found for temporal falsification.")
        logger.warning(f"Available years: {[int(y) for y in unique_years]}")
        logger.warning("Temporal falsification requires historical CHR data. Skipping this test.")
        return pd.DataFrame()
    
    # Prefer last 4 years up to 2022
    years = sorted(cand_years)[-4:] if len(cand_years) >= 4 else cand_years
    logger.info(f"Using years for temporal falsification: {years}")

    # Prepare centered moderators (assumed 2023-level adoption as provided in vcm)
    df = base_df.copy()
    df = ensure_centered(df, MO1_FAMILY + [MO2])

    # Pop by year
    if pop_long_c3.empty:
        logger.warning("Population data (chunk_3) empty; using base population for all years.")
        # Fallback: use base population
        pop = pd.DataFrame({
            'county_fips': df['county_fips'],
            'year': years[0],  # dummy year
            'log_pop_y': df['log_population']
        })
    else:
        pop = pop_long_c3[["county_fips","year_num","population_raw_value"]].dropna()
        pop["log_pop_y"] = np.log(pd.to_numeric(pop["population_raw_value"], errors="coerce").clip(lower=1))
        pop = pop.rename(columns={"year_num":"year"}).drop(columns=["population_raw_value"])
        pop = pop[pop["log_pop_y"].notna()]

    rows = []
    for y in years:
        logger.info(f"Processing temporal falsification for year {y}...")
        
        # Join DV21 for year y
        dv = dv21_long_c1[dv21_long_c1["year_num"]==y][["county_fips","premature_death_raw_value"]].copy()
        dv = dv.rename(columns={"premature_death_raw_value": f"{CORE_DV21}_{y}"})
        
        # Log merge stats
        logger.info(f"  DV21 records for year {y}: {len(dv)}")
        
        d = df.merge(dv, on="county_fips", how="left")
        
        # Merge population
        pop_y = pop[pop["year"]==y] if not pop.empty else pop
        d = d.merge(pop_y[["county_fips","log_pop_y"]], on="county_fips", how="left")
        
        # If no log_pop_y, use base log_population
        if "log_pop_y" not in d.columns or d["log_pop_y"].isna().all():
            d["log_pop_y"] = d["log_population"]
        
        ycol = f"{CORE_DV21}_{y}"
        
        # Count available data
        available = d[ycol].notna().sum()
        logger.info(f"  Available {ycol} records after merge: {available}")
        
        if available < 30:
            logger.warning(f"  Skipping year {y}: insufficient data (n={available})")
            continue
        
        base_cols = ["iv1_medicaid_expansion_active","iv2_physical_environment_score",
                     "iv4_social_economic_factors_score","log_pop_y"] + [c for c in d.columns if c.startswith("div_")]

        # a) Composite MO1 (original behavior, also reports MO2)
        cols = [ycol, f"{MO1}_c", f"{MO2}_c", "state_fips_for_clustering"] + base_cols
        dat = d[[c for c in cols if c in d.columns]].dropna()
        if not dat.empty:
            X = dat[[f"{MO1}_c", f"{MO2}_c"] + base_cols]
            res = run_ols_clustered(dat[ycol], X, dat["state_fips_for_clustering"])
            for t in [f"{MO1}_c", f"{MO2}_c"]:
                ci = res.conf_int().loc[t].values if t in res.params.index else (np.nan, np.nan)
                rows.append({
                    "Family":"Temporal falsification",
                    "Year": y,
                    "Treatment": t,
                    "Moderator": MO_CODE[MO1] if t.startswith(MO1) else MO_CODE[MO2],
                    "N": len(dat),
                    "Beta": res.params.get(t, np.nan),
                    "p": res.pvalues.get(t, np.nan),
                    "CI_lower": ci[0], "CI_upper": ci[1]
                })

        # b) Components MO11–MO15 (paired with MO2; record component effect)
        for m in MO1_COMPONENTS:
            if m not in d.columns: 
                continue
            cols = [ycol, f"{m}_c", f"{MO2}_c", "state_fips_for_clustering"] + base_cols
            dat = d[[c for c in cols if c in d.columns]].dropna()
            if dat.empty:
                continue
            X = dat[[f"{m}_c", f"{MO2}_c"] + base_cols]
            res = run_ols_clustered(dat[ycol], X, dat["state_fips_for_clustering"])
            t = f"{m}_c"
            ci = res.conf_int().loc[t].values if t in res.params.index else (np.nan, np.nan)
            rows.append({
                "Family":"Temporal falsification",
                "Year": y,
                "Treatment": t,
                "Moderator": MO_CODE[m],
                "N": len(dat),
                "Beta": res.params.get(t, np.nan),
                "p": res.pvalues.get(t, np.nan),
                "CI_lower": ci[0], "CI_upper": ci[1]
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = out.groupby("Year")["p"].transform(bh_fdr)
        out["Pass_Conservative"] = (out["p"] > 0.10) & (out["q"] > 0.25)
        out.to_csv(os.path.join(out_dir, "temporal_falsification_dv21.csv"), index=False)
        logger.info(f"Wrote temporal falsification results ({len(out)} rows).")
    else:
        logger.warning("No temporal falsification results generated.")
    return out

def run_permutation_test(df: pd.DataFrame,
                         base_controls: List[str],
                         out_dir: str,
                         logger,
                         n_perm: int = 500,
                         seed: int = 42) -> pd.DataFrame:
    """
    Within-state permutation tests for main moderation:
    DV21 ~ IV3_c × MOx_c + controls + FE, for MOx ∈ {MO1, MO11..MO15}
    """
    np.random.seed(seed)

    results = []
    for m in MO1_FAMILY:
        cols = [CORE_DV21, IV3, m, "state_fips_for_clustering"] + base_controls
        cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
        d = df[cols].dropna().copy()
        if d.empty:
            continue

        d[f"{IV3}_c"] = center(d[IV3])
        d[f"{m}_c"]   = center(d[m])
        term = f"{IV3}_c_x_{m}_c"

        def fit_beta(dat: pd.DataFrame) -> float:
            X = dat[[f"{IV3}_c", f"{m}_c"] + base_controls].copy()
            X[term] = dat[f"{IV3}_c"] * dat[f"{m}_c"]
            res = run_ols_clustered(dat[CORE_DV21], X, dat["state_fips_for_clustering"])
            return float(res.params.get(term, np.nan))

        # Observed
        beta_obs = fit_beta(d)

        # Permutations within state
        betas = []
        for _ in range(n_perm):
            dp = d.copy()
            dp[f"{m}_c"] = dp.groupby("state_fips_for_clustering")[f"{m}_c"].transform(
                lambda s: np.random.permutation(s.values))
            betas.append(fit_beta(dp))

        null = np.array(betas, dtype=float)
        # empirical two-sided p
        if np.isnan(beta_obs) or np.isnan(null).all():
            p_emp = np.nan
        else:
            less = np.sum(null <= beta_obs)
            greater = np.sum(null >= beta_obs)
            p_emp = 2 * min((less+1)/(len(null)+1), (greater+1)/(len(null)+1))

        # plot
        if VISUALS and np.isfinite(null).sum() > 10:
            plt.figure(figsize=(9,6))
            plt.hist(null, bins=40, alpha=0.8)
            plt.axvline(beta_obs, linewidth=2)
            title_p = f"{p_emp:.3f}" if pd.notna(p_emp) else "nan"
            plt.title(f"Permutation null (within-state) for IV3×{MO_CODE[m]} on {CORE_DV21}\nEmpirical p={title_p}")
            plt.xlabel(f"β_IV3×{MO_CODE[m]}")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"permutation_hist_iv3x{MO_CODE[m].lower()}_dv21.png"), dpi=300)
            plt.close()

        results.append({
            "Family":"Permutation RI",
            "Moderator": MO_CODE[m],
            "Spec": f"DV21 ~ IV3_c × {MO_CODE[m]}_c",
            "N_permutations": len(null),
            "Beta_observed": beta_obs,
            "Empirical_p": p_emp
        })

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(out_dir, "permutation_main_moderation.csv"), index=False)
    logger.info("Permutation tests done for MO1 family.")
    return out

def guardrail_checks(df: pd.DataFrame,
                     latest_c2: pd.DataFrame,
                     base_controls: List[str],
                     out_dir: str,
                     logger) -> pd.DataFrame:
    """
    1) Incoherent sign: MO effect on DV21 vs Life Expectancy — for MO1 and all components
    2) No-hospital counties: show interaction is not identified; report IV3->DV2 slope in no-hospital subset
    """
    rows = []

    # 1) Incoherent sign
    life = latest_c2.set_index("county_fips") if not latest_c2.empty else None
    if life is not None and LIFE_EXP in life.columns and CORE_DV21 in df.columns:
        d = df.join(life[[LIFE_EXP]], on="county_fips")
        for m in MO1_FAMILY:
            if m not in d.columns:
                continue
            dsub = d[["county_fips","state_fips_for_clustering", LIFE_EXP, CORE_DV21, m] + base_controls].dropna()
            if dsub.empty:
                continue
            # DV21 ~ MOm + controls
            res1 = run_ols_clustered(dsub[CORE_DV21], dsub[[m] + base_controls], dsub["state_fips_for_clustering"])
            b1 = res1.params.get(m, np.nan)

            # LifeExp ~ MOm + controls
            res2 = run_ols_clustered(dsub[LIFE_EXP], dsub[[m] + base_controls], dsub["state_fips_for_clustering"])
            b2 = res2.params.get(m, np.nan)

            incoherent = (pd.notna(b1) and pd.notna(b2) and (np.sign(b1) == np.sign(b2)) and (abs(b1)>0) and (abs(b2)>0))
            rows.append({
                "Check": f"Incoherent sign ({MO_CODE[m]} on DV21 vs LifeExp)",
                "Moderator": MO_CODE[m],
                "Beta_on_DV21": b1, "Beta_on_LifeExp": b2,
                "Incoherent": bool(incoherent)
            })

    # 2) No-hospital counties (unchanged logic; interaction undefined where MO==0)
    if (MO1 in df.columns) and (MO2 in df.columns):
        d0 = df[(df[MO1].fillna(0)==0) & (df[MO2].fillna(0)==0)]
        if not d0.empty and CORE_DV2 in d0.columns:
            d0 = d0[[CORE_DV2, IV3, "state_fips_for_clustering"] + base_controls].dropna()
            if not d0.empty:
                res = run_ols_clustered(d0[CORE_DV2], d0[[IV3] + base_controls], d0["state_fips_for_clustering"])
                rows.append({
                    "Check":"No-hospital counties",
                    "N_no_hospital": len(d0),
                    "Beta_IV3_on_DV2_in_no_hospital": res.params.get(IV3, np.nan),
                    "Note":"Interaction undefined (MO==0); slope reported for consistency."
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(os.path.join(out_dir, "guardrails_checks.csv"), index=False)
        logger.info("Guardrail checks written.")
    return out

# -----------------------------------------------------------------------------
# Fast shortlist (10 checks listed), expanded to components where relevant
# -----------------------------------------------------------------------------
def run_shortlist(df: pd.DataFrame,
                  latest_c3: pd.DataFrame,
                  latest_c2: pd.DataFrame,
                  dv21_long_c1: pd.DataFrame,
                  base_controls: List[str],
                  out_dir: str,
                  logger) -> pd.DataFrame:

    results = []

    # convenience joiners
    c3 = latest_c3.set_index("county_fips") if not latest_c3.empty else None
    c2 = latest_c2.set_index("county_fips") if not latest_c2.empty else None

    def join_col(d, col):
        if c3 is not None and col in c3.columns:
            return d.join(c3[[col]], on="county_fips")
        if c2 is not None and col in c2.columns:
            return d.join(c2[[col]], on="county_fips")
        return None

    base = df[["county_fips","state_fips_for_clustering","log_population",
               "iv1_medicaid_expansion_active","iv2_physical_environment_score",
               "iv3_health_behaviors_score","iv4_social_economic_factors_score",
               CORE_DV2, CORE_DV21, MO2] +
              [c for c in MO1_FAMILY if c in df.columns] +
              [c for c in df.columns if c.startswith("div_")]].copy()
    base = ensure_centered(base, [IV3] + MO1_FAMILY + [MO2])

    def _ols(ycol, Xcols, name):
        d = base[[ycol, "state_fips_for_clustering"] + Xcols].dropna()
        if d.empty: 
            return None
        res = run_ols_clustered(d[ycol], d[Xcols], d["state_fips_for_clustering"])
        key = Xcols[0]
        ci = res.conf_int().loc[key].values if key in res.params.index else (np.nan, np.nan)
        return {"Test":name, "N":len(d), "Beta":res.params.get(key, np.nan),
                "p":res.pvalues.get(key, np.nan),
                "CI_lower": ci[0], "CI_upper": ci[1]}

    # 1. Direct effect placebos (report each moderator)
    for dv in ["voter_turnout_raw_value", "traffic_volume_raw_value", "gender_pay_gap_raw_value"]:
        D = join_col(base, dv)
        if D is None: continue
        D = D.copy(); D[dv] = winsorize(D[dv])
        for m in (MO1_FAMILY + [MO2]):
            t = f"{m}_c"
            Xcols = [t] + base_controls
            test = _ols(dv, Xcols, f"Direct: {dv} ~ {MO_CODE[m]}_c + controls+FE")
            if test: results.append(test)

    # 2. Placebo interactions (replace IV3)
    for (ycol, piv) in [(CORE_DV2, "voter_turnout_raw_value"),
                        (CORE_DV21, "traffic_volume_raw_value")]:
        D = join_col(base, piv)
        if D is None: continue
        D = D.copy(); D[f"{piv}_c"] = center(D[piv])
        current_controls = [c for c in base_controls if c != 'iv3_health_behaviors_score']
        for m in (MO1_FAMILY + [MO2]):
            mo = f"{m}_c"; term = f"{piv}_c_x_{mo}"
            cols = [ycol, f"{piv}_c", mo, "state_fips_for_clustering"] + current_controls
            d = D[[c for c in cols if c in D.columns]].dropna()
            if d.empty: continue
            X = d[[f"{piv}_c", mo] + current_controls].copy()
            res = run_ols_clustered(d[ycol], X.assign(**{term: d[f"{piv}_c"]*d[mo]}), d["state_fips_for_clustering"])
            ci = res.conf_int().loc[term].values if term in res.params.index else (np.nan, np.nan)
            results.append({"Test":f"Placebo Int: {ycol} ~ {piv}_c×{MO_CODE[m]}_c",
                            "N":len(d), "Beta":res.params.get(term, np.nan),
                            "p":res.pvalues.get(term, np.nan),
                            "CI_lower":ci[0], "CI_upper":ci[1]})

    # 3. Placebo outcomes (keep IV3×MO)
    for (dv, m) in [("reading_scores_raw_value", MO1),
                     ("long_commute_driving_alone_raw_value", MO2)]:
        # For MO1 case, also include the components
        test_set = [m] + (MO1_COMPONENTS if m == MO1 else [])
        for mm in test_set:
            D = join_col(base, dv)
            if D is None: continue
            D = D.copy(); D[dv] = winsorize(D[dv])
            mo = f"{mm}_c"; term = f"{IV3}_c_x_{mo}"
            cols = [dv, f"{IV3}_c", mo, "state_fips_for_clustering"] + base_controls
            d = D[[c for c in cols if c in D.columns]].dropna()
            if d.empty: continue
            X = d[[f"{IV3}_c", mo] + base_controls].copy()
            res = run_ols_clustered(d[dv], X.assign(**{term: d[f"{IV3}_c"]*d[mo]}), d["state_fips_for_clustering"])
            ci = res.conf_int().loc[term].values if term in res.params.index else (np.nan, np.nan)
            results.append({"Test":f"Placebo DV: {dv} ~ IV3_c×{MO_CODE[mm]}_c",
                            "N":len(d), "Beta":res.params.get(term, np.nan),
                            "p":res.pvalues.get(term, np.nan),
                            "CI_lower":ci[0], "CI_upper":ci[1]})

    out = pd.DataFrame(results)
    if not out.empty:
        out["q"] = bh_fdr(out["p"])
        out["Pass_Conservative"] = (out["p"] > 0.10) & (out["q"] > 0.25)
        out.to_csv(os.path.join(out_dir, "shortlist_10_checks.csv"), index=False)
        logger.info("Shortlist checks written.")
    return out

# -----------------------------------------------------------------------------
# Diagnostic function for CHR columns
# -----------------------------------------------------------------------------
def diagnose_chr_columns(engine: Engine, logger):
    """
    Diagnostic function to check what's actually in the CHR columns we're looking for.
    Run this once to understand the data issues.
    """
    logger.info("="*60)
    logger.info("DIAGNOSTIC: Checking CHR column data quality")
    logger.info("="*60)
    
    problem_cols = [
        "long_commute_driving_alone_raw_value",
        "school_funding_adequacy_raw_value", 
        "school_segregation_raw_value"
    ]
    
    for table in ["chr_analytic_chunk_2", "chr_analytic_chunk_3"]:
        if not table_exists(engine, table):
            continue
            
        logger.info(f"\nChecking {table}...")
        existing = list_columns(engine, table)
        
        for col in problem_cols:
            if col not in existing:
                logger.info(f"  ❌ {col}: NOT FOUND in table")
                continue
                
            # Check sample values
            query = f"""
            SELECT 
                {col},
                COUNT(*) as count,
                COUNT(DISTINCT {col}) as distinct_count
            FROM public.{table}
            WHERE year = '2024'
            GROUP BY {col}
            ORDER BY count DESC
            LIMIT 10
            """
            
            log_sql(query, None, logger)
            
            try:
                df = pd.read_sql_query(text(query), engine)
                logger.info(f"  ✓ {col}: Found with {len(df)} distinct values")
                logger.info(f"    Top values: {df[col].tolist()[:5]}")
                
                # Try numeric conversion
                test_series = df[col].replace('', np.nan)
                numeric_series = pd.to_numeric(test_series, errors='coerce')
                valid_pct = (numeric_series.notna().sum() / len(df)) * 100 if len(df) > 0 else 0
                logger.info(f"    Numeric conversion: {valid_pct:.1f}% valid")
                
            except Exception as e:
                logger.error(f"  ❌ {col}: Error checking - {e}")
    
    logger.info("="*60)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    logger = setup_logger()
    out_dir = "negative_controls_output"
    os.makedirs(out_dir, exist_ok=True)

    engine = connect_to_database(logger)
    
    # DIAGNOSTIC: Run this once to see what's in the data
    # Comment out after you understand the issue
    diagnose_chr_columns(engine, logger)

    # Base and preparation
    base_df = fetch_base(engine, logger)
    base_df, base_controls = prep_common(base_df, logger)

    # Fetch CHR chunks
    # CORRECTED: Fixed table assignments based on where columns actually are
    c3_cols = list(set([
        "population_raw_value",
        "voter_turnout_raw_value",
        "census_participation_raw_value",
        "traffic_volume_raw_value",
        "gender_pay_gap_raw_value",
        "reading_scores_raw_value", 
        "math_scores_raw_value",
        "school_funding_adequacy_raw_value",   # MOVED: Actually in chunk_3
        "school_segregation_raw_value",        # MOVED: Actually in chunk_3
        "homeownership_raw_value",
        "severe_housing_cost_burden_raw_value",
        "not_proficient_in_english_raw_value",
        "female_raw_value",
        "below_18_years_of_age_raw_value",
        "_65_and_older_raw_value",
        "non_hispanic_black_raw_value",
        "hispanic_raw_value",
        "non_hispanic_white_raw_value",
        "juvenile_arrests_raw_value"
    ]))
    
    c2_cols = list(set([
        "driving_alone_to_work_raw_value",
        "long_commute_driving_alone_raw_value",  # MOVED: Actually in chunk_2
        "air_pollution_particulate_matter_raw_value",
        "drinking_water_violations_raw_value",
        "life_expectancy_raw_value",
        "severe_housing_problems_raw_value",
        "percentage_of_households_with_high_housing_costs",
        "percentage_of_households_with_overcrowding",
        "pct_households_with_lack_of_kitchen_or_plumbing",
    ]))
    
    c1_cols = ["premature_death_raw_value"]  # DV21 with year for temporal falsification

    c3_long = fetch_chr_chunk(engine, "chr_analytic_chunk_3", c3_cols, logger)
    c2_long = fetch_chr_chunk(engine, "chr_analytic_chunk_2", c2_cols, logger)
    c1_long = fetch_chr_chunk(engine, "chr_analytic_chunk_1", c1_cols, logger)

    # latest per county (for cross-sectional NCOs)
    latest_c3 = latest_by_county(c3_long) if not c3_long.empty else pd.DataFrame()
    latest_c2 = latest_by_county(c2_long) if not c2_long.empty else pd.DataFrame()

    # Respect DV21 unreliability flags if present (drop flagged in chunk_1)
    if table_exists(engine, "chr_analytic_chunk_1", logger):
        cols = list_columns(engine, "chr_analytic_chunk_1", logger)
        flag = "premature_death_flag_0_no_flag_1_unreliable_2_suppressed"
        if flag in cols and not c1_long.empty:
            # fetch flags and filter
            # FIX: Use _5_digit_fips as the key
            flag_sql = f"SELECT _5_digit_fips, year, {flag} FROM public.chr_analytic_chunk_1;"
            log_sql(flag_sql, None, logger)
            flags = pd.read_sql_query(text(flag_sql), engine)
            # Alias to county_fips for the merge
            flags = flags.rename(columns={"_5_digit_fips": "county_fips"})
            flags["county_fips"] = flags["county_fips"].astype(str).str.strip().str.zfill(5)
            flags["year_num"] = pd.to_numeric(flags["year"], errors="coerce")
            c1_long = c1_long.merge(flags[["county_fips","year_num",flag]], on=["county_fips","year_num"], how="left")
            c1_long = c1_long[(c1_long[flag].isna()) | (c1_long[flag].astype(str).isin(["0"]))].drop(columns=[flag])
    
    # Log data availability summary
    logger.info(f"Data summary:")
    logger.info(f"  Base counties: {len(base_df)}")
    logger.info(f"  CHR chunk_3: {len(latest_c3)} counties (latest year)")
    logger.info(f"  CHR chunk_2: {len(latest_c2)} counties (latest year)")
    logger.info(f"  CHR chunk_1 temporal: {len(c1_long)} county-year records")
    
    # Check if we have historical data
    if not c1_long.empty and 'year_num' in c1_long.columns:
        year_range = c1_long['year_num'].dropna()
        if len(year_range) > 0:
            logger.info(f"  Year range in chunk_1: {int(year_range.min())}-{int(year_range.max())}")
            historical_years = sum((2010 <= y <= 2022) for y in year_range.unique())
            logger.info(f"  Historical years (2010-2022) available: {historical_years}")
            if historical_years == 0:
                logger.warning("  ⚠️  No historical CHR data available - temporal falsification will be skipped")
                logger.warning("  ⚠️  To run temporal falsification, load CHR data from 2010-2022")

    # ------------------ Run test families ------------------
    direct_nco = run_direct_nco_tests(base_df, latest_c3, latest_c2, base_controls, out_dir, logger)
    placebo_iv  = run_placebo_interactions_replace_iv3(base_df, latest_c3, latest_c2, base_controls, out_dir, logger)
    placebo_dv  = run_placebo_outcomes_keep_iv3(base_df, latest_c3, latest_c2, base_controls, out_dir, logger)
    temporal    = run_temporal_falsification(engine, base_df, c1_long, c3_long, base_controls, out_dir, logger)
    permute     = run_permutation_test(base_df, base_controls, out_dir, logger, n_perm=500, seed=42)
    guardrails  = guardrail_checks(base_df, latest_c2, base_controls, out_dir, logger)

    # ------------------ Precision-of-null summary ------------------
    def prec_of_null(dfres: pd.DataFrame, fam_col="Family") -> pd.DataFrame:
        if dfres is None or dfres.empty:
            return pd.DataFrame()
        d = dfres.copy()
        # CI width and |β| vs CI width
        if "CI_lower" in d.columns and "CI_upper" in d.columns:
            d["CI_width"] = d["CI_upper"] - d["CI_lower"]
        if "Beta" in d.columns:
            d["AbsBeta_over_CIwidth"] = np.where(d.get("CI_width", np.nan).astype(float) > 0,
                                                 np.abs(d["Beta"]) / d["CI_width"], np.nan)
        return d

    prec_frames = []
    for part in [direct_nco, placebo_iv, placebo_dv, temporal]:
        if part is not None and not part.empty:
            prec_frames.append(prec_of_null(part))
    if prec_frames:
        prec = pd.concat(prec_frames, ignore_index=True)
        prec.to_csv(os.path.join(out_dir, "precision_of_null_summary.csv"), index=False)
        logger.info("Precision-of-null summary saved.")

    # ------------------ Unified index ------------------
    def wrangle_name(df, nm):
        if df is None or df.empty: return pd.DataFrame()
        out = df.copy()
        out["Section"] = nm
        return out
    unified = pd.concat([
        wrangle_name(direct_nco, "Direct NCO"),
        wrangle_name(placebo_iv, "Placebo Interactions (replace IV3)"),
        wrangle_name(placebo_dv, "Placebo Outcomes (keep IV3×MO)"),
        wrangle_name(temporal, "Temporal falsification"),
        wrangle_name(permute, "Permutation RI"),
        wrangle_name(guardrails, "Guardrails")
    ], ignore_index=True, sort=False)
    unified.to_csv(os.path.join(out_dir, "negative_controls_master_index.csv"), index=False)
    
    # Summary report
    logger.info("="*60)
    logger.info("NEGATIVE CONTROLS TEST SUITE COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: ./{out_dir}")
    logger.info(f"Tests completed:")
    logger.info(f"  ✓ Direct NCO tests: {len(direct_nco) if not direct_nco.empty else 0} results")
    logger.info(f"  ✓ Placebo interactions: {len(placebo_iv) if not placebo_iv.empty else 0} results")
    logger.info(f"  ✓ Placebo outcomes: {len(placebo_dv) if not placebo_dv.empty else 0} results")
    logger.info(f"  {'✓' if not temporal.empty else '⚠'} Temporal falsification: {len(temporal) if not temporal.empty else 0} results")
    logger.info(f"  ✓ Permutation tests: {len(permute) if not permute.empty else 0} results")
    logger.info(f"  ✓ Guardrail checks: {len(guardrails) if not guardrails.empty else 0} results")
    logger.info("="*60)

if __name__ == "__main__":
    main()