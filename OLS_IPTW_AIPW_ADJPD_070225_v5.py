#!/usr/bin/env python3
"""
This script performs a three-part analysis on US county-level health data:
1. Causal Main Effect Analysis: Estimates the impact of specific Generative AI
   and Robotics capabilities adoption in hospitals on premature death rates (YPLL).
   Methods: OLS (with state-clustered SEs), IPTW, and AIPW.
   Includes FDR (Benjamini-Hochberg) correction for AIPW ATE p-values.
   Treatment definition for tech components uses a median split.

2. OLS Moderation Analysis: Investigates the moderating effects of these AI/Robotics
   capabilities on the relationship between selected independent variables (IVs)
   and dependent health/operational outcomes (DVs), including health outcomes and
   hospital operational efficiency.
   Uses OLS with state-clustered SEs.
   Moderator definition uses a >0% adoption threshold.

3. OLS Direct Effect on Profitability: Investigates the direct association between
   AI/Robotics capabilities adoption and hospital operational efficiency (DV3).
   Uses OLS with state-clustered SEs and a >0% adoption threshold.

It iterates through:
- Individual technology components for main effect ATEs on health outcomes.
- Pre-defined IV-Moderator-DV interactions for moderation effects.
- Individual technology components for direct effects on operational efficiency.

Key outputs:
- Summary tables for all three analysis parts.
- Plots: ATE comparison, propensity scores, interaction plots, and a direct effect plot for profitability.
"""

import os
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import warnings
import traceback

from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample # For bootstrapping
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib or seaborn not found. Visualizations will be disabled.")

##############################################################################
# LOGGING SETUP
##############################################################################
def setup_logger(log_file_name_prefix="genai_robotics_health_analysis_log"):
    logger = logging.getLogger("genai_robotics_health_analysis")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_file_name_prefix}_{timestamp}.txt"
    log_dir = "logs"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(ch_formatter)

    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s")
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logging to: {log_path}")

    bootstrap_logger = logging.getLogger("bootstrap_internal")
    if not bootstrap_logger.hasHandlers():
        bootstrap_logger.setLevel(logging.WARNING)
    return logger

##############################################################################
# DATABASE CONNECTION & DATA FETCH
##############################################################################
def connect_to_database(logger):
    host = os.getenv("POSTGRES_HOST", 'localhost')
    database = os.getenv("POSTGRES_DB", 'Research_TEST')
    user = os.getenv("POSTGRES_USER", 'postgres')
    password = os.getenv("POSTGRESQL_KEY")

    if password is None:
        logger.error("POSTGRESQL_KEY environment variable not set. Cannot connect to database.")
        logger.error("Please set this variable to your PostgreSQL password.")
        sys.exit("Database password not configured. Exiting.")
        
    try:
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"
        engine = create_engine(conn_str)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL database '{database}' successfully.")
        return engine
    except Exception as e:
        logger.error(f"Database connection failed for database '{database}': {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def fetch_data_for_analysis(engine, logger):
    # MODIFIED: Added composite scores (MO1, MO2) directly to the query.
    sql_query = """
    SELECT
        vcm.county_fips,
        -- IVs for main effects and moderation
        vcm.health_behaviors_score,        -- IV3 (X for some interactions)
        vcm.social_economic_factors_score, -- IV4 (Confounder)
        vcm.physical_environment_score,    -- IV2 (Confounder)
        vcm.medicaid_expansion_active,     -- IV1 (Confounder)
        
        -- DVs for main effects and moderation
        vcm.health_outcomes_score,         -- DV2 (Y for some interactions)
        vcm.clinical_care_score,           -- DV1 
        vcm.avg_patient_services_margin,   -- DV3 

        -- Specific DV components from vw_conceptual_model_variables
        vcv.premature_death_raw_value,     -- DV21 (Y for main effect analysis, and some interactions)
        vcv.ratio_of_population_to_primary_care_physicians, -- DV12 (X for some interactions)
        vcv.preventable_hospital_stays_raw_value, -- DV15 (Y for some interactions)
        
        -- Moderators - Composite scores (MO1 & MO2)
        vcm.weighted_ai_adoption_score,    -- MO1
        vcm.weighted_robotics_adoption_score, -- MO2
        
        -- Moderators - Tech components from vw_adjpd_weighted_tech_summary
        vcts.pct_wfaiart_enabled_adjpd,     -- MO11 (AI Automate Routine Tasks)
        vcts.pct_wfaioacw_enabled_adjpd,    -- MO12 (AI Optimize Workflows)
        vcts.pct_wfaippd_enabled_adjpd,     -- MO13 (AI Predict Pt Demand)
        vcts.pct_wfaipsn_enabled_adjpd,     -- MO14 (AI Predict Staff Needs)
        vcts.pct_wfaiss_enabled_adjpd,      -- MO15 (AI Staff Scheduling)
        vcts.pct_robohos_enabled_adjpd,     -- MO21 (Robotics in Hospital)
        
        -- Controls
        vcm.population,                    -- Control / for lives saved
        vcm.census_division                -- Control
    FROM
        public.vw_conceptual_model_adjpd AS vcm
    LEFT JOIN 
        public.vw_conceptual_model_variables_adjpd AS vcv
        ON vcm.county_fips = vcv.county_fips
    LEFT JOIN
        public.vw_adjpd_weighted_tech_summary AS vcts
        ON vcm.county_fips = vcts.county_fips
    WHERE
        vcm.population IS NOT NULL AND CAST(vcm.population AS NUMERIC) > 0;
    """
    try:
        df = pd.read_sql_query(sql_query, engine)
        logger.info(f"Data for analysis retrieved: {df.shape[0]} rows, {df.shape[1]} columns.")
        if df.empty:
            logger.error("Fetched DataFrame is empty. Check query and data sources.")
            sys.exit(1)

        df['population'] = pd.to_numeric(df['population'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['population'], inplace=True) 
        df = df[df['population'] > 0]
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} rows due to non-positive or non-numeric population post-load.")

        if df.empty:
            logger.error("DataFrame is empty after initial population filtering. Cannot proceed.")
            sys.exit(1)
        
        logger.info(f"Renaming columns for clarity (e.g. iv3_health_behaviors_score)...")
        # MODIFIED: Added renaming for new columns.
        rename_map = {
            'health_behaviors_score': 'iv3_health_behaviors_score',
            'social_economic_factors_score': 'iv4_social_economic_factors_score',
            'physical_environment_score': 'iv2_physical_environment_score',
            'medicaid_expansion_active': 'iv1_medicaid_expansion_active',
            'health_outcomes_score': 'dv2_health_outcomes_score',
            'clinical_care_score': 'dv1_clinical_care_score',
            'avg_patient_services_margin': 'dv3_avg_patient_services_margin',
            'premature_death_raw_value': 'dv21_premature_death_ypll_rate',
            'ratio_of_population_to_primary_care_physicians': 'dv12_physicians_ratio',
            'preventable_hospital_stays_raw_value': 'dv15_preventable_stays_rate',
            'weighted_ai_adoption_score': 'mo1_genai_composite_score',
            'weighted_robotics_adoption_score': 'mo2_robotics_composite_score'
        }
        df.rename(columns=rename_map, inplace=True)
        
        nan_counts = df.isnull().sum()
        cols_with_many_nans = nan_counts[nan_counts > 0.1 * len(df)] 
        if not cols_with_many_nans.empty:
            logger.warning(f"Columns with >10% NaNs which might affect sample sizes:\n{cols_with_many_nans}")

        return df
    except Exception as e:
        logger.error(f"Database query or initial data processing failed: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

##############################################################################
# COMMON DATA PREPARATION
##############################################################################
def common_prepare_data(df_input, logger):
    logger.info("Performing common data preparations...")
    df = df_input.copy() 
    df.columns = [c.lower() for c in df.columns] 

    if 'county_fips' in df.columns:
        df['state_fips_for_clustering'] = df['county_fips'].astype(str).str.zfill(5).str[:2]
        logger.info("Created 'state_fips_for_clustering' column for clustered SEs.")
    else:
        logger.error("'county_fips' column not found. Cannot create state_fips for clustering. Exiting.")
        sys.exit(1)
        
    if 'population' not in df.columns or df['population'].isnull().any() or not df['population'].gt(0).all():
        logger.error("Population column is missing, has NaNs, or non-positive values. Critical error.")
        sys.exit(1)
    df['log_population'] = np.log(df['population'])

    medicaid_col = 'iv1_medicaid_expansion_active' 
    if medicaid_col in df.columns:
        if df[medicaid_col].dtype == bool or df[medicaid_col].dtype == np.bool_:
            df[medicaid_col] = df[medicaid_col].astype(int)
        elif df[medicaid_col].dtype == object:
            map_dict = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0, 1:1, 0:0}
            df[medicaid_col] = df[medicaid_col].astype(str).str.lower().map(map_dict)
        df[medicaid_col] = pd.to_numeric(df[medicaid_col], errors='coerce')
        if df[medicaid_col].isnull().any():
            logger.warning(f"NaNs found in '{medicaid_col}' after conversion. These rows might be dropped.")
    else:
        logger.warning(f"'{medicaid_col}' column not found for conversion.")

    census_dummy_cols = []
    if 'census_division' in df.columns:
        df['census_division'] = df['census_division'].astype(str)
        try:
            census_dummies = pd.get_dummies(df['census_division'], prefix='div', drop_first=True, dtype=int)
            df = pd.concat([df, census_dummies], axis=1)
            census_dummy_cols = list(census_dummies.columns)
            logger.info(f"Created {len(census_dummy_cols)} dummy variables for 'census_division'.")
        except Exception as e:
            logger.error(f"Failed to create dummy variables for census_division: {e}")
    else:
        logger.warning("'census_division' column not found. Skipping dummy variable creation.")

    base_confounder_cols = [
        'iv4_social_economic_factors_score', 
        'iv2_physical_environment_score',    
        'iv1_medicaid_expansion_active',     
        'log_population'
    ] + census_dummy_cols
    
    base_confounder_cols = [col for col in base_confounder_cols if col in df.columns]
    
    # MODIFIED: Added dv3 to the list of potential outcome columns
    potential_outcome_cols = [
        'dv21_premature_death_ypll_rate', 'dv2_health_outcomes_score',
        'dv15_preventable_stays_rate', 'dv1_clinical_care_score',
        'dv3_avg_patient_services_margin'
    ]
    for col in potential_outcome_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                logger.warning(f"NaNs introduced or present in outcome column '{col}' after numeric conversion.")
    
    # MODIFIED: Added composite scores to the list of tech columns to process
    tech_components_raw_names_lower = [
        'pct_wfaiart_enabled_adjpd', 'pct_wfaioacw_enabled_adjpd',
        'pct_wfaippd_enabled_adjpd', 'pct_wfaipsn_enabled_adjpd',
        'pct_wfaiss_enabled_adjpd',  'pct_robohos_enabled_adjpd',
        'mo1_genai_composite_score', 'mo2_robotics_composite_score'
    ]
    for tech_col in tech_components_raw_names_lower:
        if tech_col in df.columns:
            df[tech_col] = pd.to_numeric(df[tech_col], errors='coerce')
            # Fill NaNs with 0 for tech adoption scores, assuming NaN means no adoption data / 0% adoption.
            # This is a key assumption - review if it holds for current data.
            # If NaN means data is missing for a county that *does* have hospitals, this assumption is reasonable.
            # If NaN means the county has no hospitals, those rows might be filtered later anyway.
            nan_count_before = df[tech_col].isnull().sum()
            if nan_count_before > 0:
                df[tech_col].fillna(0, inplace=True)
                logger.info(f"Filled {nan_count_before} NaNs with 0 for tech column '{tech_col}'.")
        else:
            logger.warning(f"Tech component column {tech_col} not found for numeric conversion.")

    logger.info("Common data preparations complete.")
    return df, base_confounder_cols

#----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS FOR MAIN EFFECT ANALYSIS (OLS, IPTW, AIPW)
# These functions are largely unchanged but are called by the main script logic.
#----------------------------------------------------------------------------------------------------------------------

##############################################################################
# PROPENSITY SCORE ESTIMATION (Used by IPTW and AIPW for main effects)
##############################################################################
def estimate_propensity_scores(X_confounders, T_treatment, logger, treatment_name, C_param=0.1):
    scaler = StandardScaler()
    X_confounders_numeric = X_confounders.apply(pd.to_numeric, errors='coerce')
    if X_confounders_numeric.isnull().any().any():
        X_confounders_numeric = X_confounders_numeric.fillna(X_confounders_numeric.mean()) 

    if X_confounders_numeric.empty:
        logger.error(f"Confounders DataFrame is empty for {treatment_name} after numeric conversion/imputation.")
        return None, None
        
    X_scaled = scaler.fit_transform(X_confounders_numeric)
    
    try:
        if len(np.unique(T_treatment)) < 2:
            logger.error(f"Propensity score model for {treatment_name}: Treatment variable has only one class. Cannot fit.")
            return None, None
        if np.min(np.bincount(T_treatment.astype(int))) < 5 :
            logger.warning(f"Propensity score model for {treatment_name}: Very few samples in one treatment class. Results may be unstable.")

        prop_model = LogisticRegression(solver='liblinear', random_state=42, C=C_param, penalty='l1', max_iter=300)
        prop_model.fit(X_scaled, T_treatment)
        prop_scores = prop_model.predict_proba(X_scaled)[:, 1]
    except ValueError as ve:
        logger.error(f"ValueError in propensity score model fitting for {treatment_name}: {ve}. Check for NaNs or single class in T.")
        return None, None
    except Exception as e:
        logger.error(f"Propensity score model fitting failed for {treatment_name}: {e}")
        return None, None
    
    prop_scores_clamped = np.clip(prop_scores, 0.01, 0.99)
    return prop_scores_clamped, scaler

##############################################################################
# OLS ANALYSIS (for MAIN EFFECTS of tech components with CLUSTERED SEs)
##############################################################################
def run_ols_main_effect_analysis_clustered(df_analysis_input, treatment_col, outcome_col, confounder_cols, cluster_col, logger):
    logger.info(f"--- Starting OLS Main Effect Analysis (Clustered SEs) for Treatment: {treatment_col} on {outcome_col} ---")
    df_analysis = df_analysis_input.copy()
    try:
        Y = df_analysis[outcome_col]
        X_cols = [treatment_col] + confounder_cols
        X = df_analysis[X_cols].copy() 
        
        for col in X.columns:
            if X[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = sm.add_constant(X, has_constant='add') 

        model_df = pd.concat([Y, X, df_analysis[cluster_col]], axis=1).dropna()
        if model_df.empty or len(model_df) < model_df.shape[1] -1 :
             logger.warning(f"OLS Main Effect (Clustered): Insufficient data for {treatment_col} on {outcome_col} after NaN removal (N={len(model_df)}).")
             return np.nan, np.nan, np.nan, np.nan, True, len(model_df)

        Y_model = model_df[outcome_col]
        X_model = model_df[X.columns]
        clusters = model_df[cluster_col]

        if clusters.nunique() < 2:
            logger.warning(f"OLS Main Effect (Clustered): Less than 2 unique clusters for {treatment_col}. Using standard OLS.")
            model = sm.OLS(Y_model, X_model).fit()
        else:
            model = sm.OLS(Y_model, X_model).fit(cov_type='cluster', cov_kwds={'groups': clusters})
        
        beta = model.params[treatment_col]
        p_value = model.pvalues[treatment_col]
        ci = model.conf_int().loc[treatment_col]
        ci_lower, ci_upper = ci[0], ci[1]

        logger.info(f"OLS Main Effect (Clustered) Beta for {treatment_col}: {beta:.4f}, P-value: {p_value:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        return beta, p_value, ci_lower, ci_upper, False, len(model_df)
    except Exception as e:
        logger.error(f"OLS main effect analysis (clustered) failed for {treatment_col}: {e}")
        logger.debug(traceback.format_exc())
        return np.nan, np.nan, np.nan, np.nan, True, 0


# IPTW and AIPW functions (largely unchanged from original script)
def _calculate_iptw_ate_atet(Y, T, ps_clamped):
    weights_ate_t = T / ps_clamped
    weights_ate_c = (1 - T) / (1 - ps_clamped)
    
    mean_y1_ate = np.sum(Y * weights_ate_t) / np.sum(weights_ate_t) if np.sum(weights_ate_t) > 0 else np.nan
    mean_y0_ate = np.sum(Y * weights_ate_c) / np.sum(weights_ate_c) if np.sum(weights_ate_c) > 0 else np.nan
    ate = mean_y1_ate - mean_y0_ate

    mean_y1_atet = np.mean(Y[T == 1]) if sum(T == 1) > 0 else np.nan
    if sum(T == 0) > 0 and sum(ps_clamped[T==0]) > 0 : 
        weights_control_for_atet = (ps_clamped[T == 0]) / (1 - ps_clamped[T == 0])
        if np.sum(weights_control_for_atet) > 0:
             mean_y0_t1_atet = np.average(Y[T == 0], weights=weights_control_for_atet)
        else:
            mean_y0_t1_atet = np.nan 
    else:
        mean_y0_t1_atet = np.nan
    atet = mean_y1_atet - mean_y0_t1_atet
    return ate, atet

def run_iptw_main_effect_analysis(df_analysis, treatment_col, outcome_col, confounder_cols, n_bootstraps, logger):
    logger.info(f"--- Starting IPTW Main Effect Analysis for Treatment: {treatment_col} on {outcome_col} ---")
    error_flag = False
    ate_iptw, atet_iptw = np.nan, np.nan
    ate_ci_lower, ate_ci_upper = np.nan, np.nan
    atet_ci_lower, atet_ci_upper = np.nan, np.nan

    try:
        T_treatment = df_analysis[treatment_col].values.astype(int)
        Y_outcome = df_analysis[outcome_col].values
        X_confounders = df_analysis[confounder_cols].copy()

        if X_confounders.isnull().any().any():
            X_confounders = X_confounders.fillna(X_confounders.mean())

        prop_scores_clamped, _ = estimate_propensity_scores(X_confounders, T_treatment, logger, treatment_col)
        if prop_scores_clamped is None:
            logger.error(f"IPTW Main Effect: Propensity score estimation failed for {treatment_col}.")
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, True

        ate_iptw, atet_iptw = _calculate_iptw_ate_atet(Y_outcome, T_treatment, prop_scores_clamped)
        logger.info(f"IPTW Main Effect ATE for {treatment_col}: {ate_iptw:.4f}")
        
        ate_estimates_boot, atet_estimates_boot = [], []
        df_reset_for_bootstrap = df_analysis.reset_index(drop=True)

        for i in range(n_bootstraps):
            try:
                bootstrap_sample = resample(df_reset_for_bootstrap, replace=True, random_state=i)
                if bootstrap_sample.empty or bootstrap_sample[treatment_col].nunique() < 2: continue

                bs_T = bootstrap_sample[treatment_col].values.astype(int)
                bs_Y = bootstrap_sample[outcome_col].values
                bs_X_conf = bootstrap_sample[confounder_cols].copy()
                
                if bs_X_conf.isnull().values.any() or pd.Series(bs_Y).isnull().any(): continue
                bs_X_conf = bs_X_conf.fillna(bs_X_conf.mean()) 

                bs_ps_clamped, _ = estimate_propensity_scores(bs_X_conf, bs_T, logging.getLogger("bootstrap_internal"), treatment_col)
                if bs_ps_clamped is None or len(np.unique(bs_T)) < 2: continue
                
                bs_ate, bs_atet = _calculate_iptw_ate_atet(bs_Y, bs_T, bs_ps_clamped)
                if not np.isnan(bs_ate) and np.isfinite(bs_ate): ate_estimates_boot.append(bs_ate)
                if not np.isnan(bs_atet) and np.isfinite(bs_atet): atet_estimates_boot.append(bs_atet)
            except Exception: pass
        
        if ate_estimates_boot:
            ate_ci_lower, ate_ci_upper = np.percentile(ate_estimates_boot, 2.5), np.percentile(ate_estimates_boot, 97.5)
        if atet_estimates_boot:
            atet_ci_lower, atet_ci_upper = np.percentile(atet_estimates_boot, 2.5), np.percentile(atet_estimates_boot, 97.5)

    except Exception as e:
        logger.error(f"IPTW main effect analysis failed for {treatment_col}: {e}")
        logger.debug(traceback.format_exc())
        error_flag = True
    
    return ate_iptw, ate_ci_lower, ate_ci_upper, atet_iptw, atet_ci_lower, atet_ci_upper, error_flag

def _fit_predict_single_model_for_outcome(X_data, Y_data, T_condition_mask, model_name_suffix, treatment_name_logging, logger_obj):
    y_subset = Y_data[T_condition_mask]
    if y_subset.ndim > 1 and y_subset.shape[1] == 1: y_subset = y_subset.ravel()

    X_data_imputed = X_data.copy()
    if np.isnan(X_data_imputed).any():
        col_means = np.nanmean(X_data_imputed, axis=0)
        inds = np.where(np.isnan(X_data_imputed))
        X_data_imputed[inds] = np.take(col_means, inds[1])

    if sum(T_condition_mask) > X_data_imputed.shape[1] and sum(T_condition_mask) > 5 :
        model = sm.OLS(y_subset, sm.add_constant(X_data_imputed[T_condition_mask], has_constant='add')) 
        try:
            fitted_model = model.fit()
            return fitted_model.predict(sm.add_constant(X_data_imputed, has_constant='add'))
        except Exception as e:
            return np.full(len(Y_data), np.mean(y_subset) if sum(T_condition_mask) > 0 else np.mean(Y_data)) 
    else:
        return np.full(len(Y_data), np.mean(y_subset) if sum(T_condition_mask) > 0 else np.mean(Y_data)) 

def fit_outcome_models(X_confounders_scaled, T_treatment, Y_outcome, logger, treatment_name):
    mu1_hat = _fit_predict_single_model_for_outcome(X_confounders_scaled, Y_outcome, (T_treatment == 1), "mu1", treatment_name, logger)
    mu0_hat = _fit_predict_single_model_for_outcome(X_confounders_scaled, Y_outcome, (T_treatment == 0), "mu0", treatment_name, logger)
    return mu0_hat, mu1_hat

def _calculate_aipw_ate_atet(T_treatment, Y_outcome, prop_scores_clamped, mu0_hat, mu1_hat, logger, treatment_name):
    term1_ate = (T_treatment / prop_scores_clamped) * (Y_outcome - mu1_hat) + mu1_hat
    term0_ate = ((1 - T_treatment) / (1 - prop_scores_clamped)) * (Y_outcome - mu0_hat) + mu0_hat
    
    ate_aipw = np.mean(term1_ate[np.isfinite(term1_ate)]) - np.mean(term0_ate[np.isfinite(term0_ate)]) if (np.isfinite(term1_ate).any() and np.isfinite(term0_ate).any()) else np.nan

    if sum(T_treatment == 1) > 0:
        mean_y_t1 = np.mean(Y_outcome[T_treatment == 1])
        mean_mu0_t1 = np.mean(mu0_hat[T_treatment == 1])
        atet_aipw = mean_y_t1 - mean_mu0_t1
    else:
        atet_aipw = np.nan
    return ate_aipw, atet_aipw

def run_aipw_main_effect_analysis(df_analysis, treatment_col, outcome_col, confounder_cols, n_bootstraps, logger, plot_dir):
    logger.info(f"--- Starting AIPW Main Effect Analysis for Treatment: {treatment_col} on {outcome_col} ---")
    ate_aipw, atet_aipw = np.nan, np.nan
    ate_ci_lower, ate_ci_upper = np.nan, np.nan
    atet_ci_lower, atet_ci_upper = np.nan, np.nan
    aipw_ate_p_value = np.nan 
    
    T_treatment = df_analysis[treatment_col].values.astype(int)
    Y_outcome = df_analysis[outcome_col].values
    X_confounders = df_analysis[confounder_cols].copy()

    n_treated_overall = sum(T_treatment == 1)
    n_control_overall = sum(T_treatment == 0)
    control_outcome_mean = Y_outcome[T_treatment == 0].mean() if n_control_overall > 0 else np.nan
    control_outcome_sd = Y_outcome[T_treatment == 0].std() if n_control_overall > 1 else np.nan

    if X_confounders.isnull().any().any():
        X_confounders = X_confounders.fillna(X_confounders.mean())

    prop_scores_clamped, scaler_obj = estimate_propensity_scores(X_confounders, T_treatment, logger, treatment_col)
    if prop_scores_clamped is None or scaler_obj is None:
        logger.error(f"AIPW Main Effect: Propensity score estimation failed for {treatment_col}.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n_treated_overall, n_control_overall, True, control_outcome_mean, control_outcome_sd

    if VISUALIZATION_AVAILABLE:
        plt.figure(figsize=(10, 6))
        sns.histplot(prop_scores_clamped[T_treatment == 1], color="dodgerblue", label=f"Treated (N={n_treated_overall})", stat="density", common_norm=False, kde=True, bins=30, alpha=0.6)
        sns.histplot(prop_scores_clamped[T_treatment == 0], color="orangered", label=f"Control (N={n_control_overall})", stat="density", common_norm=False, kde=True, bins=30, alpha=0.6)
        plt.title(f"Propensity Score Distribution for {treatment_col} (Main Effect Analysis)"); plt.xlabel("Propensity Score"); plt.ylabel("Density"); plt.legend(); plt.tight_layout()
        ps_plot_path = os.path.join(plot_dir, f"propensity_overlap_maineffect_{treatment_col.replace('%','pct').replace(':','_')}.png")
        try: plt.savefig(ps_plot_path, dpi=300); plt.close()
        except Exception as e: logger.error(f"Failed to save PS overlap plot for {treatment_col}: {e}")

    X_confounders_scaled = scaler_obj.transform(X_confounders.apply(pd.to_numeric, errors='coerce')) 
    
    mu0_hat, mu1_hat = fit_outcome_models(X_confounders_scaled, T_treatment, Y_outcome, logger, treatment_col)
    if np.isnan(mu0_hat).all() or np.isnan(mu1_hat).all():
        logger.error(f"AIPW Main Effect: Outcome model fitting failed critically for {treatment_col}.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n_treated_overall, n_control_overall, True, control_outcome_mean, control_outcome_sd

    ate_aipw, atet_aipw = _calculate_aipw_ate_atet(T_treatment, Y_outcome, prop_scores_clamped, mu0_hat, mu1_hat, logger, treatment_col)
    logger.info(f"AIPW Main Effect ATE for {treatment_col}: {ate_aipw:.4f}")

    ate_estimates_boot, atet_estimates_boot = [], []
    df_reset_for_bootstrap = df_analysis.reset_index(drop=True)

    for i in range(n_bootstraps):
        try:
            bootstrap_sample = resample(df_reset_for_bootstrap, replace=True, random_state=i)
            if bootstrap_sample.empty or bootstrap_sample[treatment_col].nunique() < 2: continue

            bs_T = bootstrap_sample[treatment_col].values.astype(int)
            bs_Y = bootstrap_sample[outcome_col].values
            bs_X_conf = bootstrap_sample[confounder_cols].copy()

            if bs_X_conf.isnull().values.any() or pd.Series(bs_Y).isnull().any(): continue
            bs_X_conf = bs_X_conf.fillna(bs_X_conf.mean()) 

            bs_ps_clamped, bs_scaler = estimate_propensity_scores(bs_X_conf, bs_T, logging.getLogger("bootstrap_internal"), treatment_col)
            if bs_ps_clamped is None or bs_scaler is None or len(np.unique(bs_T)) < 2: continue
            
            bs_X_conf_scaled = bs_scaler.transform(bs_X_conf.apply(pd.to_numeric, errors='coerce'))
            bs_mu0_hat, bs_mu1_hat = fit_outcome_models(bs_X_conf_scaled, bs_T, bs_Y, logging.getLogger("bootstrap_internal"), treatment_col)
            if np.isnan(bs_mu0_hat).all() or np.isnan(bs_mu1_hat).all(): continue

            bs_ate, bs_atet = _calculate_aipw_ate_atet(bs_T, bs_Y, bs_ps_clamped, bs_mu0_hat, bs_mu1_hat, logging.getLogger("bootstrap_internal"), treatment_col)
            if not np.isnan(bs_ate) and np.isfinite(bs_ate): ate_estimates_boot.append(bs_ate)
            if not np.isnan(bs_atet) and np.isfinite(bs_atet): atet_estimates_boot.append(bs_atet)
        except Exception: pass

    if ate_estimates_boot:
        ate_ci_lower, ate_ci_upper = np.percentile(ate_estimates_boot, 2.5), np.percentile(ate_estimates_boot, 97.5)
        valid_boot_estimates_for_pval = np.array([est for est in ate_estimates_boot if np.isfinite(est)])
        if len(valid_boot_estimates_for_pval) > 0:
            num_valid_boot_for_pval = len(valid_boot_estimates_for_pval)
            if pd.notnull(ate_aipw) and np.isfinite(ate_aipw):
                if ate_aipw > 0:
                    count_le_zero = np.sum(valid_boot_estimates_for_pval <= 0)
                    aipw_ate_p_value = 2 * (count_le_zero + 1) / (num_valid_boot_for_pval + 1)
                elif ate_aipw < 0:
                    count_ge_zero = np.sum(valid_boot_estimates_for_pval >= 0)
                    aipw_ate_p_value = 2 * (count_ge_zero + 1) / (num_valid_boot_for_pval + 1)
                elif ate_aipw == 0: 
                    aipw_ate_p_value = 1.0
                if pd.notnull(aipw_ate_p_value): 
                    aipw_ate_p_value = min(aipw_ate_p_value, 1.0)
    
    if atet_estimates_boot:
        atet_ci_lower, atet_ci_upper = np.percentile(atet_estimates_boot, 2.5), np.percentile(atet_estimates_boot, 97.5)
    
    return (ate_aipw, ate_ci_lower, ate_ci_upper, aipw_ate_p_value, 
            atet_aipw, atet_ci_lower, atet_ci_upper, 
            n_treated_overall, n_control_overall, False, control_outcome_mean, control_outcome_sd)

##############################################################################
# E-VALUE CALCULATION (for main effects)
##############################################################################
def calculate_e_value(ate_or_beta, outcome_sd_control, outcome_mean_control=None, logger=None):
    if np.isnan(ate_or_beta) or np.isnan(outcome_sd_control) or outcome_sd_control == 0:
        if logger: logger.debug(f"E-value calculation skipped: ATE/Beta ({ate_or_beta}), SD_control ({outcome_sd_control}) invalid.")
        return np.nan
    standardized_effect_size = abs(ate_or_beta) / outcome_sd_control
    if standardized_effect_size == 0: return 1.0
    rr_approx = np.exp(0.91 * standardized_effect_size) 
    if rr_approx <= 1: return 1.0 
    try:
        e_value = rr_approx + np.sqrt(rr_approx * (rr_approx - 1))
    except ValueError: 
        if logger: logger.warning(f"E-value: Math error with rr_approx={rr_approx:.3f}")
        return np.nan
    if logger: logger.debug(f"E-value calculated: {e_value:.2f} (from ATE/Beta={ate_or_beta:.3f}, SMD={standardized_effect_size:.3f}, RR_approx={rr_approx:.3f})")
    return e_value

##############################################################################
# SCENARIO SIMULATION (LIVES SAVED - for main effects)
##############################################################################
def simulate_lives_saved_main_effect(df_population_data_for_sim, binary_treatment_col_name,
                                     effect_estimate, population_col, logger, outcome_is_ypll_rate=True,
                                     ypll_per_death_assumption=29.0):
    total_ypll_averted, total_deaths_averted = 0.0, 0.0
    if not outcome_is_ypll_rate: 
        logger.debug(f"  Lives saved simulation for main effect only applicable if outcome is YPLL rate. Skipped for {binary_treatment_col_name}.")
        return 0.0, 0.0
    if effect_estimate >= 0 or np.isnan(effect_estimate): 
        logger.debug(f"  Effect estimate ({effect_estimate:.4f}) is not beneficial or NaN for YPLL reduction. Lives saved calculation skipped.")
        return total_ypll_averted, total_deaths_averted

    ypll_rate_reduction_per_100k = -effect_estimate 
    non_adopter_mask = (df_population_data_for_sim[binary_treatment_col_name] == 0)
    non_adopter_counties_df = df_population_data_for_sim.loc[non_adopter_mask]

    if non_adopter_counties_df.empty:
        logger.info(f"  No non-adopter counties found for '{binary_treatment_col_name}' (main effect). Lives saved: 0.")
    else:
        valid_pop_sim = non_adopter_counties_df[population_col][pd.notnull(non_adopter_counties_df[population_col]) & (non_adopter_counties_df[population_col] > 0)]
        if not valid_pop_sim.empty:
            ypll_averted_values = (ypll_rate_reduction_per_100k / 100000.0) * valid_pop_sim
            total_ypll_averted = ypll_averted_values.sum()
            total_deaths_averted = total_ypll_averted / ypll_per_death_assumption if ypll_per_death_assumption != 0 else 0.0
            logger.info(f"  Scenario for non-adopters of '{binary_treatment_col_name}' (Main Effect: {effect_estimate:.4f} on YPLL rate):")
            logger.info(f"    Total YPLL potentially averted (main effect): {total_ypll_averted:,.0f}")
            logger.info(f"    Total premature deaths potentially averted (main effect): {total_deaths_averted:,.0f}")
    return total_ypll_averted, total_deaths_averted

##############################################################################
# VISUALIZATION FUNCTION (for MAIN EFFECT ATEs comparison)
##############################################################################
component_origin_order_main_effects = []

def generate_main_effect_ate_comparison_plot(results_df, plot_dir, logger):
    if not VISUALIZATION_AVAILABLE: return
    if results_df.empty:
        logger.info("Main effect results DataFrame is empty, skipping ATE comparison plot.")
        return
        
    plot_data = []
    for _, row in results_df.iterrows():
        if pd.notnull(row['OLS Coefficient (Beta)']) and pd.notnull(row['OLS CI_Lower']) and pd.notnull(row['OLS CI_Upper']):
            plot_data.append({'Component': row['Component'], 'Method': 'OLS (Clustered SE)', 'Effect': row['OLS Coefficient (Beta)'], 'CI_Lower': row['OLS CI_Lower'], 'CI_Upper': row['OLS CI_Upper']})
        if pd.notnull(row['IPTW ATE']) and pd.notnull(row['IPTW CI_Lower']) and pd.notnull(row['IPTW CI_Upper']):
            plot_data.append({'Component': row['Component'], 'Method': 'IPTW', 'Effect': row['IPTW ATE'], 'CI_Lower': row['IPTW CI_Lower'], 'CI_Upper': row['IPTW CI_Upper']})
        if pd.notnull(row['AIPW ATE']) and pd.notnull(row['AIPW CI_Upper']) and pd.notnull(row['AIPW CI_Lower']):
            plot_data.append({'Component': row['Component'], 'Method': 'AIPW', 'Effect': row['AIPW ATE'], 'CI_Lower': row['AIPW CI_Lower'], 'CI_Upper': row['AIPW CI_Upper']})
    
    if not plot_data:
        logger.info("No valid data to plot for main effect ATE comparison.")
        return

    plot_df = pd.DataFrame(plot_data)
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except: plt.style.use('ggplot')

    global component_origin_order_main_effects
    plot_df['Component'] = pd.Categorical(plot_df['Component'], categories=component_origin_order_main_effects, ordered=True)
    plot_df.sort_values('Component', inplace=True)

    fig_height = max(6, plot_df['Component'].nunique() * 1.0) 
    plt.figure(figsize=(12, fig_height))

    plot_df['err_lower'] = plot_df['Effect'] - plot_df['CI_Lower']
    plot_df['err_upper'] = plot_df['CI_Upper'] - plot_df['Effect']
    
    component_mapping = {name: i for i, name in enumerate(component_origin_order_main_effects)}
    plot_df['y_pos_base'] = plot_df['Component'].map(component_mapping).astype(float) 

    method_offsets = {'OLS (Clustered SE)': -0.2, 'IPTW': 0, 'AIPW': 0.2}
    plot_df['y_pos'] = plot_df['y_pos_base'] + plot_df['Method'].map(method_offsets)
    colors = {'OLS (Clustered SE)': 'blue', 'IPTW': 'green', 'AIPW': 'red'}

    for method, mdf in plot_df.groupby('Method'):
        plt.errorbar(x=mdf['Effect'], y=mdf['y_pos'], xerr=[mdf['err_lower'], mdf['err_upper']],
                     label=method, fmt='o', color=colors[method], capsize=3, markersize=5, elinewidth=1.5, markerfacecolor='white')

    plt.yticks(ticks=list(component_mapping.values()), labels=list(component_mapping.keys()))
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.xlabel("Estimated Main Effect on Premature Death Rate (YPLL per 100k population)\nLower is better (ATE < 0)")
    plt.ylabel("Technology Component")
    plt.title("Comparison of Estimated Main Treatment Effects (Tech Components)", fontsize=15, fontweight='bold')
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='major', linestyle=':', linewidth=0.5, axis='x')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plot_path = os.path.join(plot_dir, "Main_Effect_ATE_Comparison_Plot.png")
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight'); plt.close()
        logger.info(f"Main effect ATE comparison plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save main effect ATE comparison plot: {e}")

#----------------------------------------------------------------------------------
# FUNCTIONS FOR MODERATION & DV3 DIRECT EFFECT ANALYSIS
#----------------------------------------------------------------------------------

##############################################################################
# OLS MODERATION ANALYSIS (with CLUSTERED SEs)
##############################################################################
def run_ols_moderation_analysis_clustered(df_analysis_input, Y_col, X_col, M_binary_col, Interaction_term_col,
                                          confounder_cols, cluster_col, logger, interaction_name):
    logger.info(f"--- Starting OLS Moderation Analysis (Clustered SEs) for: {interaction_name} ---")
    df_analysis = df_analysis_input.copy()
    
    try:
        model_vars_for_ols = [Y_col, X_col, M_binary_col, Interaction_term_col] + confounder_cols
        X_interaction_model_df = df_analysis[model_vars_for_ols].copy()

        for col in X_interaction_model_df.columns:
            if X_interaction_model_df[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(X_interaction_model_df[col]):
                X_interaction_model_df[col] = pd.to_numeric(X_interaction_model_df[col], errors='coerce')
        
        model_df_clustered = pd.concat([X_interaction_model_df, df_analysis[cluster_col]], axis=1).dropna()
        
        if model_df_clustered.empty or len(model_df_clustered) < (len(confounder_cols) + 4):
            logger.warning(f"OLS Moderation (Clustered) ({interaction_name}): Insufficient data after NaN handling (N={len(model_df_clustered)}). Skipping.")
            return np.nan, np.nan, np.nan, np.nan, True, len(model_df_clustered)

        Y_model = model_df_clustered[Y_col]
        X_model_cols = [X_col, M_binary_col, Interaction_term_col] + confounder_cols
        X_model = model_df_clustered[X_model_cols]
        X_model = sm.add_constant(X_model, has_constant='add')
        clusters = model_df_clustered[cluster_col]
        
        n_obs_model = len(Y_model)

        if clusters.nunique() < 2:
            logger.warning(f"OLS Moderation (Clustered) ({interaction_name}): Less than 2 unique clusters. Using standard OLS.")
            model = sm.OLS(Y_model, X_model).fit()
        else:
            model = sm.OLS(Y_model, X_model).fit(cov_type='cluster', cov_kwds={'groups': clusters})
        
        interaction_beta = model.params[Interaction_term_col]
        interaction_p_value = model.pvalues[Interaction_term_col]
        interaction_ci = model.conf_int().loc[Interaction_term_col]
        interaction_ci_lower, interaction_ci_upper = interaction_ci[0], interaction_ci[1]

        logger.info(f"OLS Moderation (Clustered) Results for {Interaction_term_col} in {interaction_name}:")
        logger.info(f"  Interaction Beta: {interaction_beta:.4f}, P-value: {interaction_p_value:.4f}, 95% CI: [{interaction_ci_lower:.4f}, {interaction_ci_upper:.4f}]")
        
        return interaction_beta, interaction_p_value, interaction_ci_lower, interaction_ci_upper, False, n_obs_model

    except Exception as e:
        logger.error(f"OLS moderation analysis (clustered) failed for {interaction_name}: {e}")
        logger.debug(traceback.format_exc())
        return np.nan, np.nan, np.nan, np.nan, True, 0


##############################################################################
# SIMULATE LIVES SAVED DUE TO MODERATION EFFECT
##############################################################################
def simulate_lives_saved_moderation(df_population_data, X_col, M_binary_col, interaction_beta,
                                    population_col, logger, interaction_name,
                                    outcome_is_ypll_rate=True, ypll_per_death_assumption=29.0):
    if not outcome_is_ypll_rate:
        logger.debug(f"Lives saved (moderation) for {interaction_name} skipped: outcome not YPLL rate.")
        return 0.0, 0.0
    if np.isnan(interaction_beta):
        logger.debug(f"Lives saved (moderation) for {interaction_name} skipped: interaction_beta is NaN.")
        return 0.0, 0.0
    
    df_sim = df_population_data[df_population_data[M_binary_col] == 0].copy() 
    if df_sim.empty:
        logger.info(f"  No counties with {M_binary_col}=0 for {interaction_name}. Lives saved (moderation) calc: 0.")
        return 0.0, 0.0
    
    if X_col not in df_sim.columns or population_col not in df_sim.columns:
        logger.error(f"  Missing columns for lives saved (moderation) for {interaction_name}. Need {X_col}, {population_col}.")
        return 0.0, 0.0

    df_sim['ypll_rate_change_moderation'] = interaction_beta * df_sim[X_col]
    
    df_sim['ypll_averted_moderation'] = -df_sim.apply(
        lambda row: (row['ypll_rate_change_moderation'] / 100000.0) * row[population_col] if row['ypll_rate_change_moderation'] < 0 and pd.notnull(row[population_col]) and row[population_col] > 0 else 0,
        axis=1
    )
    df_sim['ypll_increased_moderation'] = df_sim.apply(
        lambda row: (row['ypll_rate_change_moderation'] / 100000.0) * row[population_col] if row['ypll_rate_change_moderation'] > 0 and pd.notnull(row[population_col]) and row[population_col] > 0 else 0,
        axis=1
    )

    total_ypll_averted = df_sim['ypll_averted_moderation'].sum()
    total_ypll_increased = df_sim['ypll_increased_moderation'].sum()
    net_ypll_impact = total_ypll_averted - total_ypll_increased 
    total_deaths_averted = total_ypll_averted / ypll_per_death_assumption if ypll_per_death_assumption != 0 else 0.0
    total_deaths_increased = total_ypll_increased / ypll_per_death_assumption if ypll_per_death_assumption != 0 else 0.0

    logger.info(f"  Scenario for counties adopting moderator for '{interaction_name}' (Interaction Beta: {interaction_beta:.4f}):")
    logger.info(f"    Total YPLL potentially AVERTED due to moderation: {total_ypll_averted:,.0f}")
    logger.info(f"    Net YPLL impact (Averted - Increased): {net_ypll_impact:,.0f}")
    logger.info(f"    Net Premature Deaths Impact (Averted - Increased): {total_deaths_averted - total_deaths_increased:,.0f}")
            
    return net_ypll_impact, total_deaths_averted - total_deaths_increased


##############################################################################
# VISUALIZE MODERATION EFFECT
##############################################################################
def plot_interaction_effect(df_plot_input, X_col, Y_col, M_binary_col, interaction_name,
                            ols_interaction_beta, ols_p_value,
                            plot_dir, logger, significance_threshold=0.10):
    if not VISUALIZATION_AVAILABLE:
        logger.info(f"Visualization disabled. Skipping interaction plot for {interaction_name}.")
        return
    if pd.isna(ols_interaction_beta) or pd.isna(ols_p_value):
        logger.info(f"Skipping interaction plot for {interaction_name} due to NaN beta or p-value.")
        return

    logger.info(f"Generating interaction plot for {interaction_name} (p={ols_p_value:.3f}).")
    df_plot = df_plot_input.copy()
    
    df_plot_clean = df_plot[[X_col, Y_col, M_binary_col]].dropna().copy()
    if df_plot_clean.empty or df_plot_clean[M_binary_col].nunique() < 2 :
        logger.warning(f"Not enough data or moderator levels to plot interaction for {interaction_name}.")
        return

    plt.figure(figsize=(10, 7))
    
    moderator_mapping = {0: f'No Adoption', 1: f'Adoption'}
    df_plot_clean['Moderator_Status'] = df_plot_clean[M_binary_col].map(moderator_mapping)

    sns.scatterplot(data=df_plot_clean, x=X_col, y=Y_col, hue='Moderator_Status', alpha=0.3, s=30, legend=False)
    
    colors = {moderator_mapping[0]: 'red', moderator_mapping[1]: 'green'}
    line_labels = {}
    
    for m_status_val, group_data in df_plot_clean.groupby(M_binary_col):
        m_status_label = moderator_mapping[m_status_val]
        if len(group_data) > 1: 
            X_sub = sm.add_constant(group_data[X_col])
            Y_sub = group_data[Y_col]
            try:
                model_sub = sm.OLS(Y_sub, X_sub).fit()
                slope_sub = model_sub.params[X_col]
                label_text = f'{m_status_label} (Slope: {slope_sub:.2f})'
            except:
                label_text = m_status_label

            sns.regplot(data=group_data, x=X_col, y=Y_col, scatter=False, 
                        label=label_text, 
                        color=colors[m_status_label], ci=95)
            line_labels[m_status_label] = label_text
            
    plt.title(f"Interaction: {interaction_name.replace('_',' ')}\nIV: {X_col}, Mod: {M_binary_col.replace('_adopted_gt_thresh','')}, DV: {Y_col}\nInt. Beta: {ols_interaction_beta:.3f}, p-val: {ols_p_value:.3f}",
              fontsize=11)
    plt.xlabel(X_col, fontsize=10)
    plt.ylabel(Y_col, fontsize=10)
    
    handles, _ = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles=handles, labels=list(line_labels.values()), title='Moderator Status', loc='best', fontsize=9)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = f"interaction_plot_{interaction_name.replace(' ', '_').replace('*','x').replace('>','to').replace(':','_')}.png"
    plot_path = os.path.join(plot_dir, plot_filename)
    try:
        plt.savefig(plot_path, dpi=300)
        plt.close() 
        logger.info(f"Interaction plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save interaction plot for {interaction_name}: {e}")
        logger.debug(traceback.format_exc())

# NEW: Visualization function for DV3 direct effects
##############################################################################
# VISUALIZE DIRECT EFFECTS ON OPERATIONAL EFFICIENCY (DV3)
##############################################################################
def generate_direct_effect_on_dv3_plot(results_df, plot_dir, logger):
    if not VISUALIZATION_AVAILABLE: return
    if results_df.empty:
        logger.info("Direct effects on DV3 results DataFrame is empty, skipping plot.")
        return

    plot_df = results_df.dropna(subset=['OLS Coefficient (Beta)', 'OLS CI_Lower', 'OLS CI_Upper']).copy()
    if plot_df.empty:
        logger.info("No valid data to plot for direct effects on DV3.")
        return

    # Sort by the effect size for better visualization
    plot_df = plot_df.sort_values('OLS Coefficient (Beta)', ascending=False)
    
    plt.figure(figsize=(10, 8))
    
    y_pos = np.arange(len(plot_df))
    plt.errorbar(x=plot_df['OLS Coefficient (Beta)'], y=y_pos, 
                 xerr=[plot_df['OLS Coefficient (Beta)'] - plot_df['OLS CI_Lower'], plot_df['OLS CI_Upper'] - plot_df['OLS Coefficient (Beta)']],
                 fmt='o', color='darkcyan', ecolor='lightgray', elinewidth=3, capsize=0, markersize=8)

    plt.yticks(y_pos, plot_df['Component'])
    plt.axvline(0, color='red', linestyle='--', linewidth=0.8)
    
    plt.xlabel("Association with Hospital Patient Services Margin (DV3)\n(OLS Coefficient, State-Clustered SEs)")
    plt.ylabel("Technology Capability")
    plt.title("Direct Association of Tech Adoption with Operational Efficiency", fontsize=15, fontweight='bold')
    plt.grid(True, which='major', linestyle=':', linewidth=0.5, axis='x')
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, "Direct_Effect_on_DV3_Plot.png")
    try:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Direct effect on DV3 plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save direct effect on DV3 plot: {e}")


##############################################################################
# MAIN SCRIPT EXECUTION
##############################################################################
def main():
    logger = setup_logger()
    logger.info("Starting GenAI/Robotics Causal and Moderation Analysis Script.")
    
    output_dir = "genai_robotics_health_output" 
    plot_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.linear_model._logistic')
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.PerfectSeparationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning) 

    # --- Parameters ---
    n_bootstraps = 500  
    moderator_binarization_threshold = 0.0 
    ypll_per_death_assumption = 29.0
    main_effect_outcome_col = 'dv21_premature_death_ypll_rate' 

    # MODIFIED: List of raw tech adoption columns (from dataframe after cleaning)
    tech_components_raw_names = [
        'pct_wfaiart_enabled_adjpd', 'pct_wfaioacw_enabled_adjpd',
        'pct_wfaippd_enabled_adjpd', 'pct_wfaipsn_enabled_adjpd',
        'pct_wfaiss_enabled_adjpd',  'pct_robohos_enabled_adjpd'
    ]
    tech_component_friendly_names_map = {
        'pct_wfaiart_enabled_adjpd': 'MO11_AI_Automate_Tasks',
        'pct_wfaioacw_enabled_adjpd': 'MO12_AI_Optimize_Workflows',
        'pct_wfaippd_enabled_adjpd': 'MO13_AI_Predict_Demand',
        'pct_wfaipsn_enabled_adjpd': 'MO14_AI_Predict_Staff',
        'pct_wfaiss_enabled_adjpd': 'MO15_AI_Staff_Schedule',
        'pct_robohos_enabled_adjpd': 'MO21_Robotics_In_Hospital',
        'mo1_genai_composite_score': 'MO1_GenAI_Composite',
        'mo2_robotics_composite_score': 'MO2_Robotics_Composite'
    }
    global component_origin_order_main_effects 
    component_origin_order_main_effects = [tech_component_friendly_names_map.get(tc, tc) for tc in tech_components_raw_names]
    
    # NEW: Configuration for direct effects on DV3
    direct_effects_on_dv3_config = [
        {'tech_col': 'mo1_genai_composite_score'},
        {'tech_col': 'mo2_robotics_composite_score'},
        {'tech_col': 'pct_wfaiart_enabled_adjpd'},
        {'tech_col': 'pct_wfaioacw_enabled_adjpd'},
        {'tech_col': 'pct_wfaippd_enabled_adjpd'},
        {'tech_col': 'pct_wfaipsn_enabled_adjpd'},
        {'tech_col': 'pct_wfaiss_enabled_adjpd'},
    ]

    # MODIFIED: Expanded interaction list with new DV3 tests
    interactions_to_test_config = [
        # Original tests on DV2 (Health Outcomes)
        {'name': 'IV3_x_MO11_on_DV2',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaiart_enabled_adjpd',  'Y_col': 'dv2_health_outcomes_score'},
        {'name': 'IV3_x_MO12_on_DV2',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaioacw_enabled_adjpd', 'Y_col': 'dv2_health_outcomes_score'},
        {'name': 'IV3_x_MO13_on_DV2',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaippd_enabled_adjpd',  'Y_col': 'dv2_health_outcomes_score'},
        {'name': 'IV3_x_MO14_on_DV2',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaipsn_enabled_adjpd',  'Y_col': 'dv2_health_outcomes_score'},
        {'name': 'IV3_x_MO15_on_DV2',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaiss_enabled_adjpd',   'Y_col': 'dv2_health_outcomes_score'},
        {'name': 'IV3_x_MO2_on_DV2',   'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'mo2_robotics_composite_score',  'Y_col': 'dv2_health_outcomes_score'}, # Using composite
        {'name': 'IV3_x_MO1_on_DV2',   'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'mo1_genai_composite_score',  'Y_col': 'dv2_health_outcomes_score'}, # Using composite

        # Original tests on DV21 (Premature Death)
        {'name': 'IV3_x_MO11_on_DV21', 'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaiart_enabled_adjpd',  'Y_col': 'dv21_premature_death_ypll_rate'},
        {'name': 'IV3_x_MO12_on_DV21', 'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaioacw_enabled_adjpd', 'Y_col': 'dv21_premature_death_ypll_rate'},
        
        # NEW: Moderation tests on DV3 (Operational Efficiency)
        {'name': 'IV3_x_MO1_on_DV3',   'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'mo1_genai_composite_score',    'Y_col': 'dv3_avg_patient_services_margin'},
        {'name': 'IV3_x_MO2_on_DV3',   'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'mo2_robotics_composite_score',   'Y_col': 'dv3_avg_patient_services_margin'},
        {'name': 'IV3_x_MO11_on_DV3',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaiart_enabled_adjpd',    'Y_col': 'dv3_avg_patient_services_margin'},
        {'name': 'IV3_x_MO12_on_DV3',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaioacw_enabled_adjpd',   'Y_col': 'dv3_avg_patient_services_margin'},
        {'name': 'IV3_x_MO13_on_DV3',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaippd_enabled_adjpd',    'Y_col': 'dv3_avg_patient_services_margin'},
        {'name': 'IV3_x_MO14_on_DV3',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaipsn_enabled_adjpd',    'Y_col': 'dv3_avg_patient_services_margin'},
        {'name': 'IV3_x_MO15_on_DV3',  'X_col': 'iv3_health_behaviors_score', 'M_pct_col': 'pct_wfaiss_enabled_adjpd',     'Y_col': 'dv3_avg_patient_services_margin'},
    ]
    
    try:
        engine = connect_to_database(logger)
        df_full_raw = fetch_data_for_analysis(engine, logger)
        if df_full_raw.empty: sys.exit(1)

        df_common_prepared, base_confounder_cols = common_prepare_data(df_full_raw, logger)
        if df_common_prepared.empty: sys.exit(1)
            
        tech_component_medians = {}
        for tech_col in tech_components_raw_names:
            if tech_col in df_common_prepared.columns and df_common_prepared[tech_col].notna().any():
                tech_component_medians[tech_col] = df_common_prepared[tech_col].median()
            else:
                logger.warning(f"Median for {tech_col} could not be calculated (all NaNs or column missing). Will skip this tech in main effects.")
                tech_component_medians[tech_col] = np.nan

    except Exception as e_setup:
        logger.error(f"Critical error during data setup: {e_setup}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # === PART 1: MAIN EFFECT ANALYSIS (OLS, IPTW, AIPW for each tech component on HEALTH OUTCOME) ===
    logger.info("="*30 + " PART 1: MAIN EFFECT ANALYSIS ON HEALTH OUTCOME " + "="*30)
    logger.info(f"Main effect outcome: {main_effect_outcome_col}")
    logger.info(f"Treatment definition for main effects: Adoption > Median for that technology.")
    all_main_effect_results_list = []

    for tech_col_raw_name in tech_components_raw_names:
        friendly_name = tech_component_friendly_names_map.get(tech_col_raw_name, tech_col_raw_name)
        logger.info(f"========== PROCESSING MAIN EFFECT FOR COMPONENT: {friendly_name} ({tech_col_raw_name}) ==========")
        current_main_effect_results = {'Component': friendly_name, 'Raw_Treatment_Col': tech_col_raw_name}

        df_current_main_effect_iter = df_common_prepared.copy()
        
        current_median = tech_component_medians.get(tech_col_raw_name)
        if pd.isna(current_median):
            logger.error(f"  Skipping main effect for {friendly_name}: Median could not be determined.")
            all_main_effect_results_list.append({**current_main_effect_results, 'Error_Message': "Median Undetermined"})
            continue

        binary_treatment_col = f"{tech_col_raw_name}_adopted_gt_median"
        df_current_main_effect_iter[binary_treatment_col] = (df_current_main_effect_iter[tech_col_raw_name] > current_median).astype(int)
        logger.info(f"  Median for {tech_col_raw_name}: {current_median:.4f}. Treatment defined as > Median.")

        other_tech_confounders = [tc for tc in tech_components_raw_names if tc != tech_col_raw_name and tc in df_current_main_effect_iter.columns]
        current_confounder_cols_final = list(set(base_confounder_cols + other_tech_confounders + ['iv3_health_behaviors_score']))
        current_confounder_cols_final = [c for c in current_confounder_cols_final if c in df_current_main_effect_iter.columns]

        cols_for_this_run = [main_effect_outcome_col, binary_treatment_col, 'population', 'state_fips_for_clustering'] + current_confounder_cols_final
        df_analytical_sample_main_effect = df_current_main_effect_iter[list(set(cols_for_this_run))].dropna()
        
        n_total = len(df_analytical_sample_main_effect)
        n_treated = df_analytical_sample_main_effect[binary_treatment_col].sum() if binary_treatment_col in df_analytical_sample_main_effect else 0
        n_control = n_total - n_treated
        current_main_effect_results.update({'N_Total':n_total, 'N_Treated': n_treated, 'N_Control': n_control, 'Median_Used': current_median})

        min_samples_per_group = max(len(current_confounder_cols_final) // 2 +1 , 10) 
        if n_total == 0 or n_treated < min_samples_per_group or n_control < min_samples_per_group or 'state_fips_for_clustering' not in df_analytical_sample_main_effect.columns:
            msg = f"  Main Effect: Insufficient samples or missing cluster_col for {friendly_name}. N={n_total}, T={n_treated}, C={n_control}."
            logger.warning(msg)
            current_main_effect_results.update({'Error_Message': msg, 'Overall_Error_Flag': True})
            all_main_effect_results_list.append(current_main_effect_results)
            continue
        
        logger.info(f"  Main Effect Final N for {friendly_name}: {n_total} (Treated: {n_treated}, Control: {n_control}) on Outcome: {main_effect_outcome_col}")
        current_main_effect_results['Overall_Error_Flag'] = False

        # OLS Main Effect (Clustered)
        ols_beta, ols_p, ols_cl, ols_cu, ols_err, _ = run_ols_main_effect_analysis_clustered(
            df_analytical_sample_main_effect, binary_treatment_col, main_effect_outcome_col, current_confounder_cols_final, 'state_fips_for_clustering', logger
        )
        current_main_effect_results.update({'OLS Coefficient (Beta)': ols_beta, 'OLS p-value': ols_p, 'OLS CI_Lower': ols_cl, 'OLS CI_Upper': ols_cu, 'OLS_Error': ols_err})
        if not ols_err and pd.notnull(ols_beta):
            ypll_ols, deaths_ols = simulate_lives_saved_main_effect(df_analytical_sample_main_effect, binary_treatment_col, ols_beta, 'population', logger, outcome_is_ypll_rate=True, ypll_per_death_assumption=ypll_per_death_assumption)
            current_main_effect_results.update({'OLS YPLL Averted': ypll_ols, 'OLS Deaths Averted': deaths_ols})
        
        # IPTW Main Effect
        iptw_ate, iptw_ate_cl, iptw_ate_cu, iptw_atet, iptw_atet_cl, iptw_atet_cu, iptw_err = run_iptw_main_effect_analysis(
            df_analytical_sample_main_effect, binary_treatment_col, main_effect_outcome_col, current_confounder_cols_final, n_bootstraps, logger
        )
        current_main_effect_results.update({'IPTW ATE': iptw_ate, 'IPTW CI_Lower': iptw_ate_cl, 'IPTW CI_Upper': iptw_ate_cu, 'IPTW ATET': iptw_atet, 'IPTW ATET_CI_Lower': iptw_atet_cl, 'IPTW ATET_CI_Upper': iptw_atet_cu, 'IPTW_Error': iptw_err})
        if not iptw_err and pd.notnull(iptw_ate):
            control_sd_iptw = df_analytical_sample_main_effect[main_effect_outcome_col][df_analytical_sample_main_effect[binary_treatment_col]==0].std()
            current_main_effect_results['IPTW Significant (95% CI)'] = "Yes" if (pd.notnull(iptw_ate_cl) and pd.notnull(iptw_ate_cu) and iptw_ate_cl * iptw_ate_cu > 0 and iptw_ate_cl < iptw_ate_cu) else "No"
            if current_main_effect_results['IPTW Significant (95% CI)'] == "Yes": current_main_effect_results['IPTW E_Value'] = calculate_e_value(iptw_ate, control_sd_iptw, logger=logger)
            ypll_iptw, deaths_iptw = simulate_lives_saved_main_effect(df_analytical_sample_main_effect, binary_treatment_col, iptw_ate, 'population', logger, outcome_is_ypll_rate=True, ypll_per_death_assumption=ypll_per_death_assumption)
            current_main_effect_results.update({'IPTW YPLL Averted': ypll_iptw, 'IPTW Deaths Averted': deaths_iptw})

        # AIPW Main Effect
        (aipw_ate, aipw_ate_cl, aipw_ate_cu, aipw_ate_p_value, 
         aipw_atet, aipw_atet_cl, aipw_atet_cu, 
         _, _, aipw_err, aipw_ctrl_mean, aipw_ctrl_sd) = run_aipw_main_effect_analysis(
            df_analytical_sample_main_effect, binary_treatment_col, main_effect_outcome_col, current_confounder_cols_final, n_bootstraps, logger, plot_dir
        )
        current_main_effect_results.update({
            'AIPW ATE': aipw_ate, 'AIPW CI_Lower': aipw_ate_cl, 'AIPW CI_Upper': aipw_ate_cu, 
            'AIPW ATE P-Value (raw)': aipw_ate_p_value, 
            'AIPW_Error': aipw_err
        })
        if not aipw_err and pd.notnull(aipw_ate):
            current_main_effect_results['AIPW Significant (95% CI)'] = "Yes" if (pd.notnull(aipw_ate_cl) and pd.notnull(aipw_ate_cu) and aipw_ate_cl * aipw_ate_cu > 0 and aipw_ate_cl < aipw_ate_cu) else "No"
            if current_main_effect_results['AIPW Significant (95% CI)'] == "Yes": current_main_effect_results['AIPW E_Value'] = calculate_e_value(aipw_ate, aipw_ctrl_sd, aipw_ctrl_mean, logger=logger)
            ypll_aipw, deaths_aipw = simulate_lives_saved_main_effect(df_analytical_sample_main_effect, binary_treatment_col, aipw_ate, 'population', logger, outcome_is_ypll_rate=True, ypll_per_death_assumption=ypll_per_death_assumption)
            current_main_effect_results.update({'AIPW YPLL Averted': ypll_aipw, 'AIPW Deaths Averted': deaths_aipw})
        
        current_main_effect_results['Overall_Error_Flag'] = current_main_effect_results.get('Overall_Error_Flag', False) or ols_err or iptw_err or aipw_err
        all_main_effect_results_list.append(current_main_effect_results)

    if all_main_effect_results_list:
        main_effects_results_df = pd.DataFrame(all_main_effect_results_list)
        if 'AIPW ATE P-Value (raw)' in main_effects_results_df.columns:
            raw_p_values_aipw = main_effects_results_df['AIPW ATE P-Value (raw)'].copy()
            valid_p_values_mask = raw_p_values_aipw.notna()
            p_values_for_fdr = raw_p_values_aipw[valid_p_values_mask].tolist()

            if p_values_for_fdr:
                fdr_alpha = 0.05 
                try:
                    rejected_fdr, q_values_fdr = fdrcorrection(p_values_for_fdr, alpha=fdr_alpha, method='indep', is_sorted=False)
                    main_effects_results_df['AIPW ATE Q-Value (BH)'] = np.nan
                    main_effects_results_df.loc[valid_p_values_mask, 'AIPW ATE Q-Value (BH)'] = np.array(q_values_fdr)
                except Exception as fdr_exc:
                    logger.error(f"FDR correction failed: {fdr_exc}")
            else:
                 main_effects_results_df['AIPW ATE Q-Value (BH)'] = np.nan
        
        main_effect_cols_ordered = [
            'Component', 'N_Total', 'OLS Coefficient (Beta)', 'OLS p-value', 'OLS CI_Lower', 'OLS CI_Upper', 'OLS Deaths Averted',
            'AIPW ATE', 'AIPW CI_Lower', 'AIPW CI_Upper', 'AIPW ATE P-Value (raw)', 'AIPW ATE Q-Value (BH)', 'AIPW Deaths Averted',
            'AIPW E_Value'
        ]
        for col in main_effect_cols_ordered:
            if col not in main_effects_results_df.columns: main_effects_results_df[col] = np.nan
        main_effects_results_df = main_effects_results_df[main_effect_cols_ordered]

        logger.info("\n--- Main Effects on Health Outcome Summary Table ---")
        logger.info("\n" + main_effects_results_df.to_string(index=False))
        main_effects_csv_path = os.path.join(output_dir, "main_effects_on_health_summary.csv")
        main_effects_results_df.to_csv(main_effects_csv_path, index=False, float_format='%.4f')
        logger.info(f"Main effects on health summary table saved to {main_effects_csv_path}")
        if VISUALIZATION_AVAILABLE:
             generate_main_effect_ate_comparison_plot(pd.DataFrame(all_main_effect_results_list), plot_dir, logger)
    else:
        logger.info("No main effect results to summarize.")


    # === PART 2: OLS MODERATION ANALYSIS (with State-Clustered SEs) ===
    logger.info("="*30 + " PART 2: OLS MODERATION ANALYSIS (State-Clustered SEs) " + "="*30)
    logger.info(f"Moderator definition for interactions: Adoption > {moderator_binarization_threshold*100}% for the specific technology.")
    all_moderation_results_list = []

    for interaction_config in interactions_to_test_config:
        interaction_name, X_col, M_pct_col, Y_col = interaction_config.values()
        
        logger.info(f"========== PROCESSING MODERATION: {interaction_name} ==========")
        current_moderation_results = {'Interaction_Name': interaction_name, 'IV': X_col, 'Moderator_Pct_Col': M_pct_col, 'DV': Y_col}
        df_current_interaction_iter = df_common_prepared.copy()

        required_cols_for_interaction = [X_col, M_pct_col, Y_col, 'population', 'state_fips_for_clustering'] + base_confounder_cols
        if any(c not in df_current_interaction_iter.columns for c in [X_col, M_pct_col, Y_col]):
            logger.error(f"  Missing core columns for interaction '{interaction_name}'. Skipping.")
            continue
            
        M_binary_col = f"{M_pct_col}_adopted_gt_thresh"
        df_current_interaction_iter[M_binary_col] = (df_current_interaction_iter[M_pct_col] > moderator_binarization_threshold).astype(int)
        Interaction_term_col = f"{X_col}_x_{M_binary_col}"
        df_current_interaction_iter[Interaction_term_col] = df_current_interaction_iter[X_col] * df_current_interaction_iter[M_binary_col]

        other_tech_confounders_for_moderation = [tc for tc in tech_components_raw_names if tc != M_pct_col and tc in df_current_interaction_iter.columns]
        confounders_for_ols_moderation = list(set(base_confounder_cols + other_tech_confounders_for_moderation))
        confounders_for_ols_moderation = [c for c in confounders_for_ols_moderation if c in df_current_interaction_iter.columns and c not in [X_col, M_binary_col, Interaction_term_col, Y_col]]

        (interaction_beta, interaction_p, interaction_cl, interaction_cu, ols_mod_err, n_obs_model) = run_ols_moderation_analysis_clustered(
            df_current_interaction_iter, Y_col, X_col, M_binary_col, Interaction_term_col,
            confounders_for_ols_moderation, 'state_fips_for_clustering', logger, interaction_name
        )
        current_moderation_results.update({
            'N_Obs_Model': n_obs_model, 'Interaction_Beta': interaction_beta, 'Interaction_P_Value': interaction_p,
            'Interaction_CI_Lower': interaction_cl, 'Interaction_CI_Upper': interaction_cu, 'OLS_Interaction_Error': ols_mod_err
        })

        outcome_is_ypll = (Y_col == main_effect_outcome_col)
        if not ols_mod_err and pd.notnull(interaction_beta) and outcome_is_ypll:
            sim_df_cols = list(set([Y_col, X_col, M_binary_col, 'population'] + confounders_for_ols_moderation))
            df_for_sim = df_current_interaction_iter[sim_df_cols].dropna()
            if not df_for_sim.empty:
                net_ypll_mod, net_deaths_mod = simulate_lives_saved_moderation(
                    df_for_sim, X_col, M_binary_col, interaction_beta, 'population',
                    logger, interaction_name, outcome_is_ypll_rate=True, ypll_per_death_assumption=ypll_per_death_assumption)
                current_moderation_results.update({'Net_YPLL_Impact_Moderation': net_ypll_mod, 'Net_Deaths_Impact_Moderation': net_deaths_mod})

        if not ols_mod_err and VISUALIZATION_AVAILABLE:
            plot_interaction_effect(df_current_interaction_iter, X_col, Y_col, M_binary_col,
                                    interaction_name, interaction_beta, interaction_p,
                                    plot_dir, logger, significance_threshold=0.10)

        all_moderation_results_list.append(current_moderation_results)

    if all_moderation_results_list:
        moderation_results_df = pd.DataFrame(all_moderation_results_list)
        moderation_cols_ordered = ['Interaction_Name', 'IV', 'Moderator_Pct_Col', 'DV', 'N_Obs_Model', 'Interaction_Beta', 'Interaction_P_Value', 'Interaction_CI_Lower', 'Interaction_CI_Upper', 'Net_Deaths_Impact_Moderation']
        for col in moderation_cols_ordered:
            if col not in moderation_results_df.columns: moderation_results_df[col] = np.nan
        moderation_results_df = moderation_results_df[moderation_cols_ordered]
        
        logger.info("\n--- Moderation Analysis Summary Table ---")
        logger.info("\n" + moderation_results_df.to_string(index=False))
        moderation_csv_path = os.path.join(output_dir, "moderation_analysis_summary.csv")
        moderation_results_df.to_csv(moderation_csv_path, index=False, float_format='%.4f')
        logger.info(f"Moderation analysis summary table saved to {moderation_csv_path}")
    else:
        logger.info("No moderation results to summarize.")

    # NEW: PART 3: DIRECT EFFECTS ON OPERATIONAL EFFICIENCY (DV3)
    logger.info("="*30 + " PART 3: DIRECT EFFECTS ON OPERATIONAL EFFICIENCY (DV3) " + "="*30)
    logger.info(f"Outcome: dv3_avg_patient_services_margin")
    logger.info(f"Treatment definition: Adoption > {moderator_binarization_threshold*100}% for the specific technology.")
    all_direct_effects_on_dv3_list = []
    
    # Add composite scores to the list of components to test against DV3
    all_tech_components_for_dv3 = tech_components_raw_names + ['mo1_genai_composite_score', 'mo2_robotics_composite_score']

    for tech_col in set(all_tech_components_for_dv3): # Use set to avoid duplicates
        friendly_name = tech_component_friendly_names_map.get(tech_col, tech_col)
        logger.info(f"========== PROCESSING DIRECT EFFECT ON DV3 FOR: {friendly_name} ==========")
        current_dv3_results = {'Component': friendly_name, 'Tech_Col': tech_col}

        df_current_dv3_iter = df_common_prepared.copy()
        
        binary_treatment_col = f"{tech_col}_adopted_gt_thresh"
        df_current_dv3_iter[binary_treatment_col] = (df_current_dv3_iter[tech_col] > moderator_binarization_threshold).astype(int)

        other_tech_confounders = [tc for tc in all_tech_components_for_dv3 if tc != tech_col and tc in df_current_dv3_iter.columns]
        confounders_for_dv3 = list(set(base_confounder_cols + other_tech_confounders + ['iv3_health_behaviors_score']))
        confounders_for_dv3 = [c for c in confounders_for_dv3 if c in df_current_dv3_iter.columns]

        (beta, p_val, ci_low, ci_high, err, n_obs) = run_ols_main_effect_analysis_clustered(
            df_current_dv3_iter, binary_treatment_col, 'dv3_avg_patient_services_margin', 
            confounders_for_dv3, 'state_fips_for_clustering', logger
        )
        current_dv3_results.update({
            'N_Obs_Model': n_obs,
            'OLS Coefficient (Beta)': beta, 'OLS p-value': p_val,
            'OLS CI_Lower': ci_low, 'OLS CI_Upper': ci_high,
            'Error': err
        })
        all_direct_effects_on_dv3_list.append(current_dv3_results)

    if all_direct_effects_on_dv3_list:
        dv3_direct_effects_df = pd.DataFrame(all_direct_effects_on_dv3_list)
        dv3_cols_ordered = ['Component', 'N_Obs_Model', 'OLS Coefficient (Beta)', 'OLS p-value', 'OLS CI_Lower', 'OLS CI_Upper', 'Error']
        for col in dv3_cols_ordered:
            if col not in dv3_direct_effects_df.columns: dv3_direct_effects_df[col] = np.nan
        dv3_direct_effects_df = dv3_direct_effects_df[dv3_cols_ordered]

        logger.info("\n--- Direct Effects on Operational Efficiency (DV3) Summary Table ---")
        logger.info("\n" + dv3_direct_effects_df.to_string(index=False))
        dv3_direct_effects_csv_path = os.path.join(output_dir, "direct_effects_on_dv3_summary.csv")
        dv3_direct_effects_df.to_csv(dv3_direct_effects_csv_path, index=False, float_format='%.4f')
        logger.info(f"Direct effects on DV3 summary table saved to {dv3_direct_effects_csv_path}")

        if VISUALIZATION_AVAILABLE:
            generate_direct_effect_on_dv3_plot(dv3_direct_effects_df, plot_dir, logger)
    else:
        logger.info("No DV3 direct effect results to summarize.")

    logger.info("Script finished. Check logs and output directory for details.")

if __name__ == "__main__":
    main()