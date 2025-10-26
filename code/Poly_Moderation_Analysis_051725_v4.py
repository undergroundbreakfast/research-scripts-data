#!/usr/bin/env python3
"""
Polynomial Moderation Analysis Script

This script extends the traditional moderation analysis to include polynomial terms,
allowing for the detection of nonlinear moderation effects in healthcare data.

Focus paths: IV3 → MO1 → DV2 (Health Behaviors → GenAI → Healthcare Quality)

Creates:
  - Moderation plots with polynomial curves
  - Research tables with polynomial terms and effects
"""

import os
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import warnings

from sqlalchemy import create_engine, text
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures

# For plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


##############################################################################
# LOGGING SETUP
##############################################################################

def setup_logger(log_file=None):
    """Simple logger setup."""
    logger = logging.getLogger("poly_moderation_analysis")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"poly_moderation_log_{timestamp}.txt"

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(ch_formatter)

    # File handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging to: {log_path}")
    return logger


##############################################################################
# DATABASE CONNECTION & DATA FETCH
##############################################################################

def connect_to_database(logger):
    """
    Create a SQLAlchemy engine to connect to Postgres.
    Assumes your Postgres password is stored in POSTGRESQL_KEY environment var
    or you can hard-code it here (not recommended).
    """
    host = 'localhost'
    database = 'Research_TEST'
    user = 'postgres'
    # --- IMPORTANT: Replace with your actual password or ensure POSTGRESQL_KEY is set ---
    password = os.getenv("POSTGRESQL_KEY", "YOUR_PASSWORD_HERE") # Keep this flexible
    # ------------------------------------------------------------------------------------

    if password == "YOUR_PASSWORD_HERE" and os.getenv("POSTGRESQL_KEY") is None:
        logger.warning("Using placeholder password. Update connect_to_database function or set POSTGRESQL_KEY environment variable.")

    try:
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"
        engine = create_engine(conn_str)
        # Quick test
        with engine.connect() as cn:
            cn.execute(text("SELECT 1"))
        logger.info("Connected to Postgres successfully.")
        return engine
    except Exception as ex:
        logger.error(f"DB connection failed: {ex}")
        sys.exit(1)


def fetch_data_from_vw_conceptual_model(engine, logger):
    """
    Pull needed columns from vw_conceptual_model with additional data cleaning.
    """
    query = """
        SELECT
            county_fips,
            medicaid_expansion_active AS iv1_public_health_policy,
            physical_environment_score AS iv2_physical_environment,
            health_behaviors_score AS iv3_health_behaviors,
            social_economic_factors_score AS iv4_social_economic_factors,
            weighted_ai_adoption_score AS mo1_genai_capabilities,
            weighted_robotics_adoption_score AS mo2_robotics_capabilities,
            clinical_care_score AS dv1_healthcare_access,
            health_outcomes_score AS dv2_healthcare_quality,
            avg_patient_services_margin AS dv3_operational_efficiency,
            population,
            census_division
        FROM public.vw_conceptual_model
        WHERE county_fips IS NOT NULL
          AND medicaid_expansion_active IS NOT NULL
          AND physical_environment_score IS NOT NULL
          AND health_behaviors_score IS NOT NULL
          AND social_economic_factors_score IS NOT NULL
          AND weighted_ai_adoption_score IS NOT NULL
          AND weighted_robotics_adoption_score IS NOT NULL
          AND clinical_care_score IS NOT NULL
          AND health_outcomes_score IS NOT NULL
          AND population IS NOT NULL
          AND census_division IS NOT NULL
    """
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Data fetched from vw_conceptual_model: shape={df.shape}")
        
        numeric_cols = [
            'iv1_public_health_policy', 'iv2_physical_environment', 
            'iv3_health_behaviors', 'iv4_social_economic_factors',
            'mo1_genai_capabilities', 'mo2_robotics_capabilities', 
            'dv1_healthcare_access', 'dv2_healthcare_quality', 
            'dv3_operational_efficiency', 'population'
        ]
        
        logger.info("Converting columns to numeric types...")
        df = safe_convert_to_numeric(df, numeric_cols) # Use the provided safe_convert_to_numeric
        
        if df.empty:
            logger.error("Fetched DataFrame is empty AFTER applying NOT NULL filters in SQL. "
                         "Check database content and view definition.")
            sys.exit(1)

        if df.isnull().any().any(): # Check after conversion too
             logger.warning(f"Data fetch STILL contains NaNs after SQL checks and numeric conversion. Investigate view/data:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        sys.exit(1)


##############################################################################
# DATA PREPROCESSING
##############################################################################

def safe_convert_to_numeric(df, columns):
    """
    Safely convert columns to numeric types, with detailed error handling.
    (Using the robust version from the user's original script)
    """
    df_copy = df.copy() # Work on a copy
    for col in columns:
        if col not in df_copy.columns:
            # logger.warning(f"Column {col} not found in DataFrame during numeric conversion.")
            continue # Silently skip if column not present, or log it
            
        orig_dtype = df_copy[col].dtype
        
        try:
            if pd.api.types.is_object_dtype(df_copy[col]) or pd.api.types.is_string_dtype(df_copy[col]):
                converted = pd.to_numeric(df_copy[col], errors='coerce')
                # Identify values that were not NaN before but became NaN after conversion
                null_mask = converted.isna() & (~df_copy[col].isna() & df_copy[col].notnull())
                
                if null_mask.any():
                    # Attempt to clean common issues for these specific problematic values
                    problematic_series = df_copy.loc[null_mask, col].astype(str)
                    # Remove commas
                    cleaned_series = problematic_series.str.replace(',', '', regex=False)
                    # Remove percentage signs
                    cleaned_series = cleaned_series.str.replace('%', '', regex=False)
                    # Remove any other non-numeric characters (except decimal point and sign)
                    cleaned_series = cleaned_series.str.replace(r'[^\d.-]+', '', regex=True)
                    
                    # Re-apply to the original DataFrame subset
                    df_copy.loc[null_mask, col] = cleaned_series
                    
                    # Try conversion again on the whole column
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                else:
                    df_copy[col] = converted
            elif pd.api.types.is_numeric_dtype(df_copy[col]):
                # If already numeric, ensure it's a standard float/int representation
                # This can help with types like 'Int64' (nullable int) if not desired downstream
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce') 
            else:
                # For other types (boolean, datetime), attempt conversion, coercing errors
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        except Exception as e:
            # This catch-all might be too broad, but it's from the original
            # logger.error(f"ERROR: Could not convert column {col} to numeric: {e}. Coercing to NaN.")
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce') # Final attempt

    return df_copy


def create_census_division_dummies(df, logger, division_col='census_division'):
    """Create dummy variables for census divisions."""
    if division_col not in df.columns:
        logger.warning(f"No {division_col} column found; skipping dummies.")
        return df, []

    df[division_col] = df[division_col].fillna('Unknown').astype(str)

    unique_divisions = sorted(df[division_col].unique())
    if not unique_divisions or ('Unknown' in unique_divisions and len(unique_divisions) == 1):
        logger.warning(f"No valid unique values found in {division_col} for dummies; skipping dummies.")
        return df, []
    
    valid_divisions = [d for d in unique_divisions if d != 'Unknown']
    if not valid_divisions: # Only 'Unknown' was present
        logger.warning(f"Only 'Unknown' category found in {division_col}; skipping dummies.")
        return df, []

    reference_category = valid_divisions[0] # First valid category alphabetically
    logger.info(f"Using '{reference_category}' as the reference category for {division_col} dummies.")

    dummies = pd.get_dummies(df[division_col], prefix='div', drop_first=False, dtype=int)
    
    ref_col_name = f'div_{reference_category}'
    dummy_cols_to_keep = [col for col in dummies.columns if col != ref_col_name and col.startswith('div_') and col != 'div_Unknown']


    if ref_col_name not in dummies.columns and reference_category != 'Unknown': # Check if ref col was generated
         logger.warning(f"Reference category column '{ref_col_name}' not found in dummies. Dummies might be incorrect or all values were 'Unknown'.")
         # If reference category was not found, it means it might not have been present or was the only one
         # In this case, drop_first=True logic might be safer if there are other valid_divisions
         if len(valid_divisions) > 1: # if more than one valid division, but ref not found, this is odd.
            # Fallback: if ref_col_name is not in dummies, but there are other valid divisions,
            # it means the reference_category was not in the data after all.
            # This should not happen if unique_divisions was derived correctly.
            # Re-evaluate dummy_cols_to_keep if this warning appears.
            pass # For now, proceed with current dummy_cols_to_keep

    df_with_dummies = pd.concat([df, dummies[dummy_cols_to_keep]], axis=1)
    
    logger.info(f"Created {len(dummy_cols_to_keep)} dummy cols for {division_col}. Kept: {dummy_cols_to_keep}")
    return df_with_dummies, dummy_cols_to_keep


def transform_data_for_analysis(df, iv, mo, dv, transformation="raw", logger=None):
    """
    Applies the specified transformation to the data for analysis.
    (Using the version from the user's original script)
    """
    df_transformed = df.copy()
    
    col_mapping = {
        'iv': {'original': iv, 'transformed': iv},
        'mo': {'original': mo, 'transformed': mo},
        'dv': {'original': dv, 'transformed': dv}
    }
    
    if transformation == "raw":
        return df_transformed, col_mapping
    
    vars_to_transform = {'iv': iv, 'mo': mo, 'dv': dv}

    for var_key, orig_name in vars_to_transform.items():
        if orig_name not in df_transformed.columns:
            if logger: logger.warning(f"Variable {orig_name} not found in DataFrame for transformation.")
            continue

        if transformation == "zscore":
            new_name = f"{orig_name}_z"
            mean = df_transformed[orig_name].mean()
            std = df_transformed[orig_name].std()
            if np.isclose(std, 0) or pd.isna(std):
                if logger: logger.warning(f"Variable {orig_name} has zero or NaN standard deviation. Z-scores set to 0.")
                df_transformed[new_name] = 0
            else:
                df_transformed[new_name] = (df_transformed[orig_name] - mean) / std
            col_mapping[var_key]['transformed'] = new_name
        
        elif transformation == "log":
            new_name = f"{orig_name}_log"
            min_val = df_transformed[orig_name].min()
            if df_transformed[orig_name].isnull().all():
                 if logger: logger.warning(f"Variable {orig_name} is all NaN. Log transform results in all NaN.")
                 df_transformed[new_name] = np.nan
            elif min_val <= 0:
                offset = abs(min_val) + 1 if min_val <=0 else 0 # Add 1 if non-positive values exist
                if logger: logger.info(f"Log transform: Adding offset of {offset} to {orig_name} (min={min_val:.4f}).")
                df_transformed[new_name] = np.log(df_transformed[orig_name] + offset)
            else:
                df_transformed[new_name] = np.log(df_transformed[orig_name])
            col_mapping[var_key]['transformed'] = new_name
        
        elif transformation == "winsor":
            new_name = f"{orig_name}_win"
            p05 = df_transformed[orig_name].quantile(0.05)
            p95 = df_transformed[orig_name].quantile(0.95)
            if pd.isna(p05) or pd.isna(p95): # Happens if too few non-NaN values
                if logger: logger.warning(f"Could not calculate quantiles for {orig_name} (possibly too many NaNs). Winsorized column will be NaN.")
                df_transformed[new_name] = np.nan
            else:
                df_transformed[new_name] = df_transformed[orig_name].clip(lower=p05, upper=p95)
                if logger: logger.info(f"Winsorized {orig_name} at 5th ({p05:.4f}) and 95th ({p95:.4f}) percentiles.")
            col_mapping[var_key]['transformed'] = new_name
        
        else:
            if var_key == 'iv': # Log error only once
                 if logger: logger.error(f"Unknown transformation: {transformation}. Using raw data.")
            # No change to df_transformed or col_mapping if transformation unknown
            return df.copy(), {k: {'original': v['original'], 'transformed': v['original']} for k,v in col_mapping.items()} # return original if error

    return df_transformed, col_mapping


##############################################################################
# POLYNOMIAL MODERATION ANALYSIS
##############################################################################

def run_polynomial_moderation(df, iv, mo, dv, controls, logger, degree=2):
    if logger: logger.info(f"Fitting polynomial moderation model: {dv} ~ poly({iv}, degree={degree}) * {mo} + controls")
    
    df_model_fit_data = df.copy() # Use this name for clarity
    
    required_cols = [iv, mo, dv] + controls
    missing_in_df = [col for col in required_cols if col not in df_model_fit_data.columns]
    if missing_in_df:
        if logger: logger.error(f"Missing required columns in DataFrame for polynomial moderation: {missing_in_df}")
        return None, df_model_fit_data # Return the original df copy

    missing_counts = df_model_fit_data[required_cols].isnull().sum()
    if missing_counts.sum() > 0:
        if logger: logger.warning(f"Missing values detected: {missing_counts[missing_counts > 0].to_dict()}")
        df_model_fit_data = df_model_fit_data.dropna(subset=required_cols)
        if logger: logger.info(f"After dropping missing values: {len(df_model_fit_data)} rows remain")
    
    if df_model_fit_data.empty:
        if logger: logger.error("DataFrame is empty after handling NaNs. Cannot fit polynomial model.")
        return None, df_model_fit_data

    n_observations = len(df_model_fit_data)
    if n_observations < (len(required_cols) + degree * 2 + 10): # Rough check for min obs
        if logger: logger.warning(f"Small sample size: {n_observations} observations for {len(required_cols) + degree*2} potential predictors. Model may be unstable.")

    # Store the names of IV and MO as they are in the input `df` (e.g., iv_t, mo_t)
    # These are the "base" variables from which centered versions will be made *within this function*
    base_iv_for_model = iv 
    base_mo_for_model = mo
    
    is_mo_binary = df_model_fit_data[base_mo_for_model].nunique() <= 2
    
    iv_centered_name = f"{base_iv_for_model}_centered"
    iv_mean = df_model_fit_data[base_iv_for_model].mean()
    if logger: logger.info(f"Mean-centering {base_iv_for_model} (mean={iv_mean:.4f}) -> {iv_centered_name}")
    df_model_fit_data[iv_centered_name] = df_model_fit_data[base_iv_for_model] - iv_mean
    
    mo_centered_name = base_mo_for_model # Default to base name if not centered
    mo_mean = 0.0 # Default for binary or uncentered
    if not is_mo_binary:
        mo_centered_name = f"{base_mo_for_model}_centered"
        mo_mean = df_model_fit_data[base_mo_for_model].mean()
        if logger: logger.info(f"Mean-centering {base_mo_for_model} (mean={mo_mean:.4f}) -> {mo_centered_name}")
        df_model_fit_data[mo_centered_name] = df_model_fit_data[base_mo_for_model] - mo_mean
    else:
        if logger: logger.info(f"{base_mo_for_model} is binary or has <=2 unique values. Not centering for model.")

    iv_array = df_model_fit_data[[iv_centered_name]].values
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    iv_poly_values = poly.fit_transform(iv_array)
    
    # Generate feature names like 'iv_centered_name', 'iv_centered_name^2'
    poly_term_names_generated = poly.get_feature_names_out([iv_centered_name])
    
    poly_df = pd.DataFrame(iv_poly_values, columns=poly_term_names_generated, index=df_model_fit_data.index)
    for col in poly_df.columns:
        df_model_fit_data[col] = poly_df[col]
    
    interaction_term_names = []
    for poly_term in poly_term_names_generated:
        inter_name = f"{poly_term}_x_{mo_centered_name}"
        df_model_fit_data[inter_name] = df_model_fit_data[poly_term] * df_model_fit_data[mo_centered_name]
        interaction_term_names.append(inter_name)
    
    X_cols = list(poly_term_names_generated) + [mo_centered_name] + interaction_term_names + controls
    # Deduplicate X_cols in case mo_centered_name was already in controls (e.g. if MO was a control)
    X_cols = sorted(list(set(X_cols)))


    X = df_model_fit_data[X_cols].copy()
    y = df_model_fit_data[dv].copy()
    
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            if logger: logger.warning(f"Column {col} in X is not numeric. Converting...")
            X[col] = pd.to_numeric(X[col], errors='coerce')
    if not pd.api.types.is_numeric_dtype(y):
        if logger: logger.warning(f"Target {dv} is not numeric. Converting...")
        y = pd.to_numeric(y, errors='coerce')
    
    combined_final = pd.concat([X, y.rename('target_y')], axis=1)
    if combined_final.isnull().any().any():
        if logger: logger.warning("Final data for model contains NaN values. Dropping affected rows.")
        mask_valid = ~combined_final.isnull().any(axis=1)
        X = X.loc[mask_valid]
        y = y.loc[mask_valid]
        if logger: logger.info(f"After dropping all NaN values from X and y: {len(X)} rows remain")

    if X.empty or y.empty or len(X) < len(X.columns) + 1:
        if logger: logger.error(f"Not enough valid data points ({len(X)}) to fit polynomial model with {len(X.columns)} predictors.")
        return None, df_model_fit_data
        
    try:
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        model.model_type = "polynomial"
        model.polynomial_degree = degree
        model.base_iv_for_model = base_iv_for_model # Name of IV col in input df (e.g., iv_t)
        model.base_mo_for_model = base_mo_for_model # Name of MO col in input df (e.g., mo_t)
        model.iv_mean_for_centering = iv_mean # Mean of base_iv_for_model used for its centering
        model.mo_mean_for_centering = mo_mean # Mean of base_mo_for_model used for its centering (0 if binary/not centered)
        
        model.iv_terms_in_model = list(poly_term_names_generated) # e.g. [iv_t_centered, iv_t_centered^2]
        model.mo_term_in_model = mo_centered_name          # e.g. mo_t_centered or mo_t (if binary)
        model.interaction_terms_in_model = interaction_term_names # Full interaction term names
        model.controls_in_model = controls
        
        if logger:
            logger.info(f"Polynomial Model Summary ({dv} ~ poly({base_iv_for_model}, degree={degree}) * {base_mo_for_model}):")
            logger.info(f"N: {model.nobs}, R²: {model.rsquared:.4f}, Adj. R²: {model.rsquared_adj:.4f}")
            coef_table = pd.DataFrame({'coef': model.params, 'std err': model.bse, 't': model.tvalues, 'P>|t|': model.pvalues})
            logger.info("Coefficients:")
            for i, row in coef_table.iterrows():
                stars = "†" if 0.05 <= row['P>|t|'] < 0.1 else ("*" * sum(row['P>|t|'] < cutoff for cutoff in (0.05, 0.01, 0.001)))
                logger.info(f"{i:<30}: {row['coef']:>10.4f} ({row['std err']:.4f}) t={row['t']:>7.3f} p={row['P>|t|']:.4f} {stars}")
            
            sig_interactions = coef_table.loc[coef_table.index.isin(interaction_term_names) & (coef_table['P>|t|'] < 0.05)]
            if not sig_interactions.empty: logger.info(f"✓ {len(sig_interactions)} significant polynomial interaction term(s) found.")
            else: logger.info("✗ No significant polynomial interaction terms found (p < 0.05).")

        X_reduced = X_with_const.drop(columns=interaction_term_names, errors='ignore')
        reduced_model = sm.OLS(y, X_reduced).fit()
        f_squared, f_squared_interp = calculate_cohens_f_squared(model.rsquared, reduced_model.rsquared)
        model.f_squared = f_squared
        model.f_squared_interpretation = f_squared_interp
        if logger: logger.info(f"Cohen's f-squared for polynomial interaction: {f_squared:.4f} ({f_squared_interp})")
        
        return model, df_model_fit_data.loc[X.index] # Return data aligned with model
    
    except Exception as e:
        if logger: logger.error(f"Error fitting polynomial moderation model: {e}")
        # import traceback
        # if logger: logger.error(traceback.format_exc())
        return None, df_model_fit_data


def calculate_cohens_f_squared(r2_full, r2_reduced):
    if pd.isna(r2_full) or pd.isna(r2_reduced): return 0.0, "Error (NaN R-squared)"
    if r2_full < r2_reduced : r2_full = r2_reduced # Should not happen if nested, but defensively
    
    f_squared = (r2_full - r2_reduced) / (1 - r2_full) if (1 - r2_full) > 1e-9 else 0.0 # Avoid division by zero if R2_full is 1
    
    if f_squared < 0.0: f_squared = 0.0 # Defensive if r2_full somehow less than r2_reduced significantly
        
    if f_squared < 0.02: interpretation = "Negligible effect"
    elif f_squared < 0.15: interpretation = "Small effect"
    elif f_squared < 0.35: interpretation = "Medium effect"
    else: interpretation = "Large effect"
    
    return f_squared, interpretation


def calculate_polynomial_conditional_effects(model, mo_value_on_base_scale, iv_points_on_base_scale):
    """
    Calculates conditional effects for polynomial moderation.
    mo_value_on_base_scale: A single value of the moderator (on its scale as input to run_polynomial_moderation, e.g. mo_t).
    iv_points_on_base_scale: An array of IV values (on its scale as input to run_polynomial_moderation, e.g. iv_t).
    """
    try:
        X_pred = pd.DataFrame()
        X_pred['const'] = 1.0 # Add constant term first

        # Center IV points using the mean of base_iv_for_model (e.g., iv_t)
        iv_centered_values = iv_points_on_base_scale - model.iv_mean_for_centering
        
        # Add polynomial terms for IV (e.g., iv_t_centered, iv_t_centered^2)
        # model.iv_terms_in_model = ['iv_t_centered', 'iv_t_centered^2']
        for term_name in model.iv_terms_in_model:
            base_iv_name_in_term = term_name.split('^')[0] # Should be model.base_iv_for_model + "_centered"
            if '^' in term_name:
                power = int(term_name.split('^')[-1])
                X_pred[term_name] = iv_centered_values ** power
            else: # Linear term
                X_pred[term_name] = iv_centered_values
        
        # Center MO value using the mean of base_mo_for_model (e.g., mo_t)
        # model.mo_mean_for_centering is 0 if MO was binary/not centered
        mo_centered_for_pred = mo_value_on_base_scale - model.mo_mean_for_centering
        X_pred[model.mo_term_in_model] = mo_centered_for_pred # model.mo_term_in_model is e.g. mo_t_centered or mo_t
        
        # Add interaction terms
        # model.interaction_terms_in_model = ['iv_t_centered_x_mo_t_centered', 'iv_t_centered^2_x_mo_t_centered']
        for inter_term_name in model.interaction_terms_in_model:
            # Expected format: poly_iv_term_x_mo_term (e.g., iv_t_centered^2_x_mo_t_centered)
            base_poly_iv_term = inter_term_name.replace(f"_x_{model.mo_term_in_model}", "")
            if base_poly_iv_term not in X_pred.columns:
                # This should not happen if model.iv_terms_in_model are correctly populated into X_pred
                print(f"ERROR: Base polynomial term '{base_poly_iv_term}' for interaction '{inter_term_name}' not found in X_pred. Columns: {X_pred.columns.tolist()}")
                return None
            X_pred[inter_term_name] = X_pred[base_poly_iv_term] * mo_centered_for_pred
        
        # Add control variables (set to 0, assuming centered or dummy with 0 as reference)
        for control in model.controls_in_model:
            if control not in X_pred.columns: # Avoid overwriting if a control is part of IV/MO terms
                 X_pred[control] = 0.0
        
        # Ensure all columns expected by the model are present and in order
        pred_cols = model.model.exog_names # Names of regressors from fitted model
        missing_cols_in_Xpred = set(pred_cols) - set(X_pred.columns)
        for col in missing_cols_in_Xpred:
            # This typically catches 'const' if not added, or if a control was missed.
            # print(f"Warning: Column '{col}' from model parameters not in X_pred. Adding as 0 for prediction.")
            X_pred[col] = 0.0 
            
        X_pred_final = X_pred[pred_cols] # Select and order columns
        
        pred_result = model.get_prediction(X_pred_final)
        summary_frame = pred_result.summary_frame(alpha=0.05) # alpha for 95% CI
        
        return {
            'mean': summary_frame['mean'].values, # Return as numpy arrays
            'ci_lower': summary_frame['mean_ci_lower'].values,
            'ci_upper': summary_frame['mean_ci_upper'].values
        }
    
    except Exception as e:
        print(f"Error calculating polynomial conditional effects: {e}")
        # import traceback
        # print(traceback.format_exc())
        # print(f"Model exog_names: {model.model.exog_names}")
        # print(f"X_pred columns before ordering: {X_pred.columns.tolist()}")
        return None


##############################################################################
# LINEAR MODERATION ANALYSIS
##############################################################################

def run_linear_moderation(df, iv, mo, dv, controls, logger):
    if logger: logger.info(f"Fitting linear moderation model: {dv} ~ {iv} * {mo} + controls")

    df_model_fit_data = df.copy()

    required_cols = [iv, mo, dv] + controls
    missing_in_df = [col for col in required_cols if col not in df_model_fit_data.columns]
    if missing_in_df:
        if logger: logger.error(f"Missing required columns in DataFrame for linear moderation: {missing_in_df}")
        return None, df_model_fit_data

    missing_counts = df_model_fit_data[required_cols].isnull().sum()
    if missing_counts.sum() > 0:
        if logger: logger.warning(f"Missing values detected: {missing_counts[missing_counts > 0].to_dict()}")
        df_model_fit_data = df_model_fit_data.dropna(subset=required_cols)
        if logger: logger.info(f"After dropping missing values: {len(df_model_fit_data)} rows remain")

    if df_model_fit_data.empty:
        if logger: logger.error("DataFrame is empty after handling NaNs. Cannot fit linear model.")
        return None, df_model_fit_data
        
    n_observations = len(df_model_fit_data)
    if n_observations < (len(required_cols) + 1 + 10): # Predictors + interaction + const + buffer
        if logger: logger.warning(f"Small sample size: {n_observations} observations for linear model. May be unstable.")

    base_iv_for_model = iv
    base_mo_for_model = mo
    
    is_mo_binary = df_model_fit_data[base_mo_for_model].nunique() <= 2
    
    iv_centered_name = f"{base_iv_for_model}_centered"
    iv_mean = df_model_fit_data[base_iv_for_model].mean()
    if logger: logger.info(f"Mean-centering {base_iv_for_model} (mean={iv_mean:.4f}) -> {iv_centered_name}")
    df_model_fit_data[iv_centered_name] = df_model_fit_data[base_iv_for_model] - iv_mean
    
    mo_centered_name = base_mo_for_model
    mo_mean = 0.0
    if not is_mo_binary:
        mo_centered_name = f"{base_mo_for_model}_centered"
        mo_mean = df_model_fit_data[base_mo_for_model].mean()
        if logger: logger.info(f"Mean-centering {base_mo_for_model} (mean={mo_mean:.4f}) -> {mo_centered_name}")
        df_model_fit_data[mo_centered_name] = df_model_fit_data[base_mo_for_model] - mo_mean
    else:
        if logger: logger.info(f"{base_mo_for_model} is binary or has <=2 unique values. Not centering for model.")

    interaction_term_name = f"{iv_centered_name}_x_{mo_centered_name}"
    df_model_fit_data[interaction_term_name] = df_model_fit_data[iv_centered_name] * df_model_fit_data[mo_centered_name]
    
    X_cols = [iv_centered_name, mo_centered_name, interaction_term_name] + controls
    X_cols = sorted(list(set(X_cols))) # Deduplicate

    X = df_model_fit_data[X_cols].copy()
    y = df_model_fit_data[dv].copy()

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            if logger: logger.warning(f"Column {col} in X is not numeric. Converting...")
            X[col] = pd.to_numeric(X[col], errors='coerce')
    if not pd.api.types.is_numeric_dtype(y):
        if logger: logger.warning(f"Target {dv} is not numeric. Converting...")
        y = pd.to_numeric(y, errors='coerce')

    combined_final = pd.concat([X, y.rename('target_y')], axis=1)
    if combined_final.isnull().any().any():
        if logger: logger.warning("Final data for model contains NaN values. Dropping affected rows.")
        mask_valid = ~combined_final.isnull().any(axis=1)
        X = X.loc[mask_valid]
        y = y.loc[mask_valid]
        if logger: logger.info(f"After dropping all NaN values from X and y: {len(X)} rows remain")

    if X.empty or y.empty or len(X) < len(X.columns) + 1:
        if logger: logger.error(f"Not enough valid data points ({len(X)}) to fit linear model with {len(X.columns)} predictors.")
        return None, df_model_fit_data
        
    try:
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        model.model_type = "linear"
        model.base_iv_for_model = base_iv_for_model
        model.base_mo_for_model = base_mo_for_model
        model.iv_mean_for_centering = iv_mean
        model.mo_mean_for_centering = mo_mean
        
        model.iv_term_in_model = iv_centered_name
        model.mo_term_in_model = mo_centered_name
        model.interaction_term_in_model = interaction_term_name
        model.controls_in_model = controls
        
        if logger:
            logger.info(f"Linear Model Summary ({dv} ~ {base_iv_for_model} * {base_mo_for_model}):")
            logger.info(f"N: {model.nobs}, R²: {model.rsquared:.4f}, Adj. R²: {model.rsquared_adj:.4f}")
            coef_table = pd.DataFrame({'coef': model.params, 'std err': model.bse, 't': model.tvalues, 'P>|t|': model.pvalues})
            logger.info("Coefficients:")
            for i, row in coef_table.iterrows():
                stars = "†" if 0.05 <= row['P>|t|'] < 0.1 else ("*" * sum(row['P>|t|'] < cutoff for cutoff in (0.05, 0.01, 0.001)))
                logger.info(f"{i:<30}: {row['coef']:>10.4f} ({row['std err']:.4f}) t={row['t']:>7.3f} p={row['P>|t|']:.4f} {stars}")
            
            interaction_p_val = coef_table.loc[interaction_term_name, 'P>|t|'] if interaction_term_name in coef_table.index else 1.0
            if interaction_p_val < 0.05: logger.info(f"✓ Significant linear interaction term found: {interaction_term_name} (p={interaction_p_val:.4f}).")
            else: logger.info(f"✗ No significant linear interaction term found (p={interaction_p_val:.4f}).")

        X_reduced = X_with_const.drop(columns=[interaction_term_name], errors='ignore')
        reduced_model = sm.OLS(y, X_reduced).fit()
        f_squared, f_squared_interp = calculate_cohens_f_squared(model.rsquared, reduced_model.rsquared)
        model.f_squared = f_squared
        model.f_squared_interpretation = f_squared_interp
        if logger: logger.info(f"Cohen's f-squared for linear interaction: {f_squared:.4f} ({f_squared_interp})")
        
        return model, df_model_fit_data.loc[X.index]
    
    except Exception as e:
        if logger: logger.error(f"Error fitting linear moderation model: {e}")
        # import traceback
        # if logger: logger.error(traceback.format_exc())
        return None, df_model_fit_data


def calculate_linear_conditional_effects(model, mo_value_on_base_scale, iv_points_on_base_scale):
    """
    Calculates conditional effects for linear moderation.
    mo_value_on_base_scale: A single value of the moderator (on its scale as input to run_linear_moderation, e.g. mo_t).
    iv_points_on_base_scale: An array of IV values (on its scale as input to run_linear_moderation, e.g. iv_t).
    """
    try:
        X_pred = pd.DataFrame()
        X_pred['const'] = 1.0

        iv_centered_values = iv_points_on_base_scale - model.iv_mean_for_centering
        X_pred[model.iv_term_in_model] = iv_centered_values # e.g., iv_t_centered
        
        mo_centered_for_pred = mo_value_on_base_scale - model.mo_mean_for_centering
        X_pred[model.mo_term_in_model] = mo_centered_for_pred # e.g., mo_t_centered or mo_t
        
        X_pred[model.interaction_term_in_model] = X_pred[model.iv_term_in_model] * X_pred[model.mo_term_in_model]
        
        for control in model.controls_in_model:
            if control not in X_pred.columns:
                 X_pred[control] = 0.0
        
        pred_cols = model.model.exog_names
        missing_cols_in_Xpred = set(pred_cols) - set(X_pred.columns)
        for col in missing_cols_in_Xpred:
            # print(f"Warning: Column '{col}' from model parameters not in X_pred. Adding as 0 for prediction.")
            X_pred[col] = 0.0
            
        X_pred_final = X_pred[pred_cols]
        
        pred_result = model.get_prediction(X_pred_final)
        summary_frame = pred_result.summary_frame(alpha=0.05)
        
        return {
            'mean': summary_frame['mean'].values,
            'ci_lower': summary_frame['mean_ci_lower'].values,
            'ci_upper': summary_frame['mean_ci_upper'].values
        }
    
    except Exception as e:
        print(f"Error calculating linear conditional effects: {e}")
        # import traceback
        # print(traceback.format_exc())
        # print(f"Model exog_names: {model.model.exog_names}")
        # print(f"X_pred columns before ordering: {X_pred.columns.tolist()}")
        return None


##############################################################################
# VISUALIZATION FUNCTIONS (ORIGINAL TWO-PANEL PLOTS)
##############################################################################

def plot_polynomial_moderation(df_plot_data, iv_col, mo_col, dv_col, model, out_path=None, logger=None):
    """
    Creates a visualization of polynomial moderation effects (scatter + 3D surface).
    df_plot_data: DataFrame containing the data (e.g., df_transformed).
    iv_col, mo_col, dv_col: Names of columns in df_plot_data (e.g., iv_t, mo_t, dv_t).
    """
    if not VISUALIZATION_AVAILABLE:
        if logger: logger.warning("Visualization libraries not found. Skipping plot_polynomial_moderation.")
        return

    fig = None  # Initialize fig to ensure it's defined for plt.close in case of early error
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 7)) 
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        # Labels based on the column names passed (e.g., iv_t, mo_t, dv_t)
        iv_label = iv_col.replace('_', ' ').title()
        dv_label = dv_col.replace('_', ' ').title()
        mo_label = mo_col.replace('_', ' ').title()

        fig.suptitle(f"Polynomial Moderation: {iv_label} × {mo_label} → {dv_label} (Degree {model.polynomial_degree})", 
                    fontsize=16, fontweight='bold', y=0.98)

        # Moderator levels for lines and point coloring are based on the `mo_col` in `df_plot_data`
        # (e.g., quantiles of mo_t)
        q33 = df_plot_data[mo_col].quantile(0.33)
        q66 = df_plot_data[mo_col].quantile(0.66)
        median_mo = df_plot_data[mo_col].median()

        # Ensure distinct values for plotting if data is sparse
        mo_plot_levels = sorted(list(set([q33, median_mo, q66])))
        if len(mo_plot_levels) == 1: # All quantiles are the same
            std_mo = df_plot_data[mo_col].std()
            if pd.isna(std_mo) or std_mo == 0 : std_mo = 1 # Failsafe
            mo_plot_levels = [median_mo - std_mo, median_mo, median_mo + std_mo]
        elif len(mo_plot_levels) == 2: # Two quantiles are the same
             mo_plot_levels = [mo_plot_levels[0], (mo_plot_levels[0]+mo_plot_levels[1])/2 , mo_plot_levels[1]]


        mo_low_val_for_line = mo_plot_levels[0]
        mo_med_val_for_line = mo_plot_levels[1] if len(mo_plot_levels) > 1 else mo_plot_levels[0]
        mo_high_val_for_line = mo_plot_levels[-1]

        if logger: 
            logger.info(f"Plotting polynomial lines for MO levels (from {mo_col}): Low={mo_low_val_for_line:.4f}, Med={mo_med_val_for_line:.4f}, High={mo_high_val_for_line:.4f}")

        # Point coloring based on quantiles of mo_col
        conditions = [
            df_plot_data[mo_col] <= q33,
            (df_plot_data[mo_col] > q33) & (df_plot_data[mo_col] <= q66),
            df_plot_data[mo_col] > q66
        ]
        choices = ['Low', 'Medium', 'High']
        # Ensure df_plot_data is a copy if we modify it
        df_scatter_plot = df_plot_data.copy()
        df_scatter_plot['mo_group_display'] = np.select(conditions, choices, default='Unknown')
        
        group_counts = df_scatter_plot['mo_group_display'].value_counts()
        if logger: logger.info(f"Point coloring groups (from {mo_col}): {group_counts.to_dict()}")
        df_scatter_plot['mo_group_display'] = pd.Categorical(df_scatter_plot['mo_group_display'], categories=['Low', 'Medium', 'High', 'Unknown'], ordered=True)
        
        # IV range for plotting lines, based on iv_col in df_plot_data
        iv_min_plot = df_plot_data[iv_col].min() 
        iv_max_plot = df_plot_data[iv_col].max()
        iv_vals_for_lines = np.linspace(iv_min_plot, iv_max_plot, 100)

        palette = {"Low": "#d62728", "Medium": "#1f77b4", "High": "#2ca02c", "Unknown": "grey"}
        line_styles = ['-', '--', '-.']
        mo_vals_for_lines_dict = {'Low': mo_low_val_for_line, 'Medium': mo_med_val_for_line, 'High': mo_high_val_for_line}

        sns.scatterplot(data=df_scatter_plot, x=iv_col, y=dv_col, hue='mo_group_display', palette=palette,
                      alpha=0.5, s=35, edgecolor='w', linewidth=0.5, ax=ax1)

        legend_lines_handles = []
        for i, (group_name, mo_val_iter) in enumerate(mo_vals_for_lines_dict.items()):
            # calculate_polynomial_conditional_effects expects mo_value and iv_points on the scale of
            # model.base_mo_for_model and model.base_iv_for_model (which are mo_col and iv_col here)
            pred_data = calculate_polynomial_conditional_effects(model, mo_val_iter, iv_vals_for_lines)
            if pred_data is None or pred_data['mean'] is None or len(pred_data['mean']) != len(iv_vals_for_lines):
                if logger: logger.warning(f"Prediction failed or returned unexpected data for MO group '{group_name}' in plot_polynomial_moderation.")
                continue

            line_color = palette.get(group_name, "black")
            mean_pred, lower_band, upper_band = pred_data['mean'], pred_data['ci_lower'], pred_data['ci_upper']
            
            ax1.fill_between(iv_vals_for_lines, lower_band, upper_band, color=line_color, alpha=0.2)
            
            slope_str = ""
            if len(iv_vals_for_lines) > 1 and len(mean_pred) == len(iv_vals_for_lines):
                slopes = np.gradient(mean_pred, iv_vals_for_lines)
                avg_slope = np.mean(slopes)
                slope_str = f"Avg slope: {avg_slope:.3f}"
            
            label = f"{group_name} {mo_label} ({slope_str})"
            line, = ax1.plot(iv_vals_for_lines, mean_pred, color=line_color, 
                          linestyle=line_styles[i % len(line_styles)], linewidth=3, label=label)
            legend_lines_handles.append(line)

        ax1.set_title('Polynomial Moderation Effect', fontsize=13)
        ax1.set_xlabel(iv_label, fontsize=12)
        ax1.set_ylabel(dv_label, fontsize=12)
        if legend_lines_handles:
            try:
                existing_handles, existing_labels = ax1.get_legend_handles_labels()
            except Exception as e:
                if logger: logger.warning(f"Could not get existing legend handles/labels: {e}")
                existing_handles, existing_labels = [], []
            
            # Combine with our line handles and their labels
            combined_handles = existing_handles + legend_lines_handles
            combined_labels = existing_labels + [h.get_label() for h in legend_lines_handles]
            
            # Create new legend with combined items
            ax1.legend(handles=combined_handles, labels=combined_labels, title=f"{mo_label} Levels")
        
        if hasattr(model, 'f_squared'):
            ax1.annotate(
                f"Cohen's f² = {model.f_squared:.4f} ({model.f_squared_interpretation})",
                xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=11, fontweight='bold'
            )

        # 3D Plot: uses iv_col, mo_col, dv_col for axes
        iv_grid_3d = np.linspace(df_plot_data[iv_col].min(), df_plot_data[iv_col].max(), 20)
        mo_grid_3d = np.linspace(df_plot_data[mo_col].min(), df_plot_data[mo_col].max(), 20)
        IV_mesh, MO_mesh = np.meshgrid(iv_grid_3d, mo_grid_3d)
        
        DV_pred_3d = np.full_like(IV_mesh, np.nan) # Initialize with NaNs
        for r_idx in range(MO_mesh.shape[0]): # iterate over mo_grid_3d values
            mo_val_for_surface = MO_mesh[r_idx, 0] # MO is constant for this row
            pred_for_surface_row = calculate_polynomial_conditional_effects(model, mo_val_for_surface, iv_grid_3d)
            if pred_for_surface_row is not None and len(pred_for_surface_row['mean']) == len(iv_grid_3d):
                 DV_pred_3d[r_idx, :] = pred_for_surface_row['mean']
        
        if not np.isnan(DV_pred_3d).all(): # Only plot if we have some valid data
            surf = ax2.plot_surface(IV_mesh, MO_mesh, DV_pred_3d, cmap='viridis', alpha=0.8, edgecolor='none')
            fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=10, pad=0.1)
        else:
            if logger: logger.warning("Could not generate 3D surface data for polynomial plot.")

        ax2.set_xlabel(iv_label, fontsize=12, labelpad=10)
        ax2.set_ylabel(mo_label, fontsize=12, labelpad=10)
        ax2.set_zlabel(dv_label, fontsize=12, labelpad=10)
        ax2.view_init(elev=20, azim=-60) 
        ax2.set_title('3D Interaction Surface', fontsize=13)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
        
        if out_path:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            if logger: logger.info(f"Saved polynomial moderation plot to: {out_path}")
        
    except Exception as e:
        if logger: logger.error(f"Error creating plot_polynomial_moderation: {e}")
        # import traceback
        # if logger: logger.error(traceback.format_exc())
    finally:
        if fig is not None and isinstance(fig, plt.Figure):
            plt.close(fig)


##############################################################################
# SIMPLIFIED VISUALIZATION FUNCTION (SINGLE PANEL)
##############################################################################

def plot_simple_moderation(df_plot_data, iv_col, mo_col, dv_col, model, is_polynomial=False, degree=1, 
                          transform_type="raw", out_path=None, logger=None):
    """
    Creates a simplified, publication-ready visualization of moderation effects.
    df_plot_data: DataFrame containing the data (e.g., df_transformed or df_original).
    iv_col, mo_col, dv_col: Names of columns in df_plot_data.
    """
    if not VISUALIZATION_AVAILABLE:
        if logger: logger.warning("Visualization libraries not found. Skipping plot_simple_moderation.")
        return

    fig = None
    try:
        sns.set_style("whitegrid")
        plt.rc('axes', titlesize=14, labelsize=12) 
        plt.rc('legend', fontsize=10)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        
        low_color, mid_color, high_color = "#d62728", "#1f77b4", "#2ca02c"
        low_style, high_style = "-", "--"
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Labels based on the column names passed, with transformation suffix
        transform_label_map = {"log": " (Log)", "winsor": " (Winsorized)", "zscore": " (Z-scored)", "raw": ""}
        transform_suffix = transform_label_map.get(transform_type, "")

        iv_label = iv_col.replace('_', ' ').title() + transform_suffix
        dv_label = dv_col.replace('_', ' ').title() + transform_suffix
        mo_label_display = mo_col.replace('_', ' ').title() + transform_suffix

        model_type_str = f"Polynomial (Degree {degree})" if is_polynomial else "Linear"
        title = f"{model_type_str} Moderation: {iv_label} × {mo_label_display} → {dv_label}"
        ax.set_title(title, fontweight='bold', fontsize=16)
        
        # Moderator levels for lines: +/- 2SD of mo_col in df_plot_data
        # Moderator levels for point coloring: +/- 1SD of mo_col in df_plot_data
        mo_mean_plot = df_plot_data[mo_col].mean() 
        mo_std_plot = df_plot_data[mo_col].std()
        if pd.isna(mo_std_plot) or mo_std_plot == 0: mo_std_plot = 1 # Failsafe

        mo_val_for_low_line = mo_mean_plot - 2 * mo_std_plot
        mo_val_for_high_line = mo_mean_plot + 2 * mo_std_plot
        if logger:
            logger.info(f"Plotting simple moderation lines for MO (from {mo_col}): Low (-2SD)={mo_val_for_low_line:.4f}, High (+2SD)={mo_val_for_high_line:.4f}")

        low_thresh_points = mo_mean_plot - mo_std_plot
        high_thresh_points = mo_mean_plot + mo_std_plot
        
        df_scatter_plot = df_plot_data.copy() # Ensure original df_plot_data is not modified
        low_mask = df_scatter_plot[mo_col] <= low_thresh_points
        high_mask = df_scatter_plot[mo_col] >= high_thresh_points
        mid_mask = (~low_mask) & (~high_mask)
        
        group_counts = {'Low MO Points': sum(low_mask), 'Medium MO Points': sum(mid_mask), 'High MO Points': sum(high_mask)}
        if logger: logger.info(f"Point coloring groups (simple plot, from {mo_col}): {group_counts}")
        
        # IV range for plotting lines, based on iv_col in df_plot_data
        iv_min_plot = df_plot_data[iv_col].min()
        iv_max_plot = df_plot_data[iv_col].max()
        iv_vals_for_lines = np.linspace(iv_min_plot, iv_max_plot, 100)
        
        calculate_effects_fn = calculate_polynomial_conditional_effects if is_polynomial else calculate_linear_conditional_effects
        
        pred_low = calculate_effects_fn(model, mo_val_for_low_line, iv_vals_for_lines)
        pred_high = calculate_effects_fn(model, mo_val_for_high_line, iv_vals_for_lines)
        
        ax.scatter(df_scatter_plot.loc[mid_mask, iv_col], df_scatter_plot.loc[mid_mask, dv_col],
                  color=mid_color, alpha=0.3, s=40, edgecolor='w', linewidth=0.5, label=f"Medium MO (±1SD of {mo_label_display})")
        ax.scatter(df_scatter_plot.loc[low_mask, iv_col], df_scatter_plot.loc[low_mask, dv_col],
                  color=low_color, alpha=0.5, s=40, edgecolor='w', linewidth=0.5, label=f"Low MO (≤ -1SD of {mo_label_display})")
        ax.scatter(df_scatter_plot.loc[high_mask, iv_col], df_scatter_plot.loc[high_mask, dv_col],
                  color=high_color, alpha=0.5, s=40, edgecolor='w', linewidth=0.5, label=f"High MO (≥ +1SD of {mo_label_display})")
        
        legend_handles_lines = []

        if pred_low is not None and pred_low['mean'] is not None and len(pred_low['mean']) == len(iv_vals_for_lines):
            mean_low, ci_low_lower, ci_low_upper = pred_low['mean'], pred_low['ci_lower'], pred_low['ci_upper']
            slope_str = ""
            if len(mean_low) > 1:
                slopes = np.gradient(mean_low, iv_vals_for_lines) if is_polynomial else [(mean_low[-1] - mean_low[0]) / (iv_vals_for_lines[-1] - iv_vals_for_lines[0]) if (iv_vals_for_lines[-1] - iv_vals_for_lines[0]) else 0]
                avg_slope = np.mean(slopes)
                slope_str = f"Avg Slope: {avg_slope:.3f}" if is_polynomial else f"Slope: {avg_slope:.3f}"
            
            ax.fill_between(iv_vals_for_lines, ci_low_lower, ci_low_upper, color=low_color, alpha=0.2)
            line_low, = ax.plot(iv_vals_for_lines, mean_low, color=low_color, linestyle=low_style, linewidth=2.5,
                               label=f"Low (-2SD) {mo_label_display} ({slope_str})")
            legend_handles_lines.append(line_low)
        else:
            if logger: logger.warning(f"Could not plot low MO line for {mo_label_display} in simple plot.")


        if pred_high is not None and pred_high['mean'] is not None and len(pred_high['mean']) == len(iv_vals_for_lines):
            mean_high, ci_high_lower, ci_high_upper = pred_high['mean'], pred_high['ci_lower'], pred_high['ci_upper']
            slope_str = ""
            if len(mean_high) > 1:
                slopes = np.gradient(mean_high, iv_vals_for_lines) if is_polynomial else [(mean_high[-1] - mean_high[0]) / (iv_vals_for_lines[-1] - iv_vals_for_lines[0]) if (iv_vals_for_lines[-1] - iv_vals_for_lines[0]) else 0]
                avg_slope = np.mean(slopes)
                slope_str = f"Avg Slope: {avg_slope:.3f}" if is_polynomial else f"Slope: {avg_slope:.3f}"

            ax.fill_between(iv_vals_for_lines, ci_high_lower, ci_high_upper, color=high_color, alpha=0.2)
            line_high, = ax.plot(iv_vals_for_lines, mean_high, color=high_color, linestyle=high_style, linewidth=2.5,
                                label=f"High (+2SD) {mo_label_display} ({slope_str})")
            legend_handles_lines.append(line_high)
        else:
            if logger: logger.warning(f"Could not plot high MO line for {mo_label_display} in simple plot.")
        
        ax.set_xlabel(iv_label, fontsize=14)
        ax.set_ylabel(dv_label, fontsize=14)
        
        # Combine legends carefully
        handles, labels = ax.get_legend_handles_labels() # Get scatter plot legend items
        # Add line legend items if they exist
        ax.legend(handles=handles + legend_handles_lines, labels=labels + [h.get_label() for h in legend_handles_lines], 
                  title=f"Moderator: {mo_label_display}", fontsize=9, title_fontsize=11, loc='best')
        
        if hasattr(model, 'f_squared'):
            fig.text(0.5, 0.015, f"Cohen's f² (Interaction) = {model.f_squared:.4f} ({model.f_squared_interpretation})", 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
            plt.subplots_adjust(bottom=0.18, top=0.92) # Adjust for title and f2 text
        else:
            plt.subplots_adjust(bottom=0.1, top=0.92)

        if out_path:
            plot_dir_name = os.path.dirname(out_path)
            if plot_dir_name and not os.path.exists(plot_dir_name):
                os.makedirs(plot_dir_name)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            if logger: logger.info(f"Saved simplified moderation plot to: {out_path}")
        
    except Exception as e:
        if logger: logger.error(f"Error creating simplified moderation plot: {e}")
        # import traceback
        # if logger: logger.error(traceback.format_exc())
    finally:
        if fig is not None and isinstance(fig, plt.Figure): 
            plt.close(fig)


##############################################################################
# RESEARCH TABLE CREATION
##############################################################################

def create_research_table_for_poly_path(iv_name_in_df, mo_name_in_df, dv_name_in_df, model, results_dir, logger, transform_type="raw", degree_val=2):
    try:
        # Use a more descriptive filename including transformation and degree
        table_name = f"poly_{transform_type}_deg{degree_val}_{model.base_iv_for_model}_x_{model.base_mo_for_model}_to_{dv_name_in_df}.txt"
        table_path = os.path.join(results_dir, table_name)
        
        with open(table_path, 'w') as f:
            f.write(f"Polynomial Moderation Analysis Results (Transform: {transform_type.upper()}, Degree: {degree_val})\n")
            f.write("="*80 + "\n")
            f.write(f"IV (base for model): {model.base_iv_for_model}\n")
            f.write(f"MO (base for model): {model.base_mo_for_model}\n")
            f.write(f"DV: {dv_name_in_df}\n")
            f.write(f"Polynomial Degree: {model.polynomial_degree}\n")
            f.write(f"N: {model.nobs}, R²: {model.rsquared:.4f}, Adj. R²: {model.rsquared_adj:.4f}\n")
            if hasattr(model, 'f_squared'):
                 f.write(f"Cohen's f² (Interaction): {model.f_squared:.4f} ({model.f_squared_interpretation})\n")
            f.write("\nCoefficients:\n")
            f.write(f"{'Term':<35} {'Coef.':>10} {'Std. Err.':>10} {'t-value':>10} {'P>|t|':>10} Sig.\n")
            f.write("-" * 80 + "\n")
            coef_table = pd.DataFrame({
                'coef': model.params, 'std err': model.bse,
                't': model.tvalues, 'P>|t|': model.pvalues
            })
            for term, row in coef_table.iterrows():
                stars = "†" if 0.05 <= row['P>|t|'] < 0.1 else ("*" * sum(row['P>|t|'] < cutoff for cutoff in (0.05, 0.01, 0.001)))
                f.write(f"{term:<35} {row['coef']:>10.4f} {row['std err']:>10.4f} {row['t']:>10.3f} {row['P>|t|']:>10.4f} {stars}\n")
        logger.info(f"Saved polynomial research table to: {table_path}")
    except Exception as e:
        logger.error(f"Failed to create polynomial research table: {e}")


def create_research_table_for_linear_path(iv_name_in_df, mo_name_in_df, dv_name_in_df, model, results_dir, logger, transform_type="raw"):
    try:
        table_name = f"linear_{transform_type}_{model.base_iv_for_model}_x_{model.base_mo_for_model}_to_{dv_name_in_df}.txt"
        table_path = os.path.join(results_dir, table_name)
        
        with open(table_path, 'w') as f:
            f.write(f"Linear Moderation Analysis Results (Transform: {transform_type.upper()})\n")
            f.write("="*80 + "\n")
            f.write(f"IV (base for model): {model.base_iv_for_model}\n")
            f.write(f"MO (base for model): {model.base_mo_for_model}\n")
            f.write(f"DV: {dv_name_in_df}\n")
            f.write(f"N: {model.nobs}, R²: {model.rsquared:.4f}, Adj. R²: {model.rsquared_adj:.4f}\n")
            if hasattr(model, 'f_squared'):
                f.write(f"Cohen's f² (Interaction): {model.f_squared:.4f} ({model.f_squared_interpretation})\n")
            f.write("\nCoefficients:\n")
            f.write(f"{'Term':<35} {'Coef.':>10} {'Std. Err.':>10} {'t-value':>10} {'P>|t|':>10} Sig.\n")
            f.write("-" * 80 + "\n")

            coef_table = pd.DataFrame({
                'coef': model.params, 'std err': model.bse,
                't': model.tvalues, 'P>|t|': model.pvalues
            })
            for term, row in coef_table.iterrows():
                stars = "†" if 0.05 <= row['P>|t|'] < 0.1 else ("*" * sum(row['P>|t|'] < cutoff for cutoff in (0.05, 0.01, 0.001)))
                f.write(f"{term:<35} {row['coef']:>10.4f} {row['std err']:>10.4f} {row['t']:>10.3f} {row['P>|t|']:>10.4f} {stars}\n")
        logger.info(f"Saved linear research table to: {table_path}")
    except Exception as e:
        logger.error(f"Failed to create linear research table: {e}")


##############################################################################
# MAIN FUNCTION WITH DATA TRANSFORMATIONS
##############################################################################

def main():
    logger = setup_logger()
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels") # Ignore some common statsmodels warnings

    output_dir = "polynomial_moderation_output"
    plot_dir = os.path.join(output_dir, "plots_twopanel") # Renamed for clarity
    simple_plot_dir = os.path.join(output_dir, "plots_simple")
    results_dir = os.path.join(output_dir, "results")
    for d in [output_dir, plot_dir, simple_plot_dir, results_dir]:
        if not os.path.exists(d): os.makedirs(d)
    logger.info(f"Output will be saved in: {output_dir}")

    engine = connect_to_database(logger)
    df_original_raw = fetch_data_from_vw_conceptual_model(engine, logger)
    if df_original_raw is None or df_original_raw.empty:
        logger.error("Failed to load data. Exiting.")
        sys.exit(1)

    df_original_raw.columns = [c.lower() for c in df_original_raw.columns]
    # Create a working copy for preprocessing that might alter structure (like adding dummies)
    df_preprocessed = df_original_raw.copy()
    df_preprocessed, div_dummies = create_census_division_dummies(df_preprocessed, logger, 'census_division')

    IV3 = "iv3_health_behaviors"
    MO1 = "mo1_genai_capabilities"
    DV2 = "dv2_healthcare_quality"
    # Base controls; specific models might add/remove some
    base_controls = ["population"] + div_dummies 
    
    # Ensure all controls actually exist in the df_preprocessed
    final_controls = [c for c in base_controls if c in df_preprocessed.columns]
    if len(final_controls) != len(base_controls):
        logger.warning(f"Some control variables were not found: {set(base_controls) - set(final_controls)}")


    all_vars_needed_initial = [IV3, MO1, DV2] + final_controls
    missing_vars_df = [v for v in all_vars_needed_initial if v not in df_preprocessed.columns]
    if missing_vars_df:
        logger.error(f"Essential variables missing from DataFrame: {', '.join(missing_vars_df)}. Exiting.")
        sys.exit(1)
    
    degrees_to_try = [2, 3]
    transformations_to_try = ["raw", "log", "winsor", "zscore"] # Added zscore to the loop
    
    for transformation in transformations_to_try:
        logger.info("="*80)
        logger.info(f"ANALYZING WITH TRANSFORMATION: {transformation.upper()}")
        logger.info("="*80)
        
        # Apply transformation using the df_preprocessed (which has dummies)
        # transform_data_for_analysis only transforms IV, MO, DV based on its internal list
        df_transformed, col_mapping = transform_data_for_analysis(
            df_preprocessed, IV3, MO1, DV2, transformation=transformation, logger=logger
        )
        
        iv_t = col_mapping['iv']['transformed']
        mo_t = col_mapping['mo']['transformed']
        dv_t = col_mapping['dv']['transformed']
        
        # Ensure transformed columns are actually in df_transformed
        if not all(c in df_transformed.columns for c in [iv_t, mo_t, dv_t]):
            logger.error(f"Transformed columns {iv_t}, {mo_t}, or {dv_t} not found after transformation '{transformation}'. Skipping.")
            continue
            
        logger.info(f"Using variables for modeling: IV={iv_t}, MO={mo_t}, DV={dv_t} (Controls: {final_controls})")
        
        # --- Polynomial Moderation Loop ---
        for degree in degrees_to_try:
            logger.info("-" * 60)
            logger.info(f"Polynomial Moderation (Degree {degree}) on {transformation} data")
            
            model_poly, df_used_poly = run_polynomial_moderation(
                df_transformed, iv_t, mo_t, dv_t, final_controls, logger, degree=degree
            )
            
            if model_poly and df_used_poly is not None and not df_used_poly.empty:
                plot_path_poly = os.path.join(plot_dir, f"poly_{transformation}_deg{degree}_{IV3}x{MO1}_{DV2}.png")
                plot_polynomial_moderation(df_used_poly, iv_t, mo_t, dv_t, model_poly, out_path=plot_path_poly, logger=logger)
                
                simple_plot_path_poly = os.path.join(simple_plot_dir, f"simple_poly_{transformation}_deg{degree}_{IV3}x{MO1}_{DV2}.png")
                plot_simple_moderation(df_used_poly, iv_t, mo_t, dv_t, model_poly, 
                                       is_polynomial=True, degree=degree, transform_type=transformation,
                                       out_path=simple_plot_path_poly, logger=logger)
                
                create_research_table_for_poly_path(iv_t, mo_t, dv_t, model_poly, results_dir, logger, 
                                                    transform_type=transformation, degree_val=degree)
            else:
                logger.warning(f"Polynomial moderation (deg {degree}, {transformation}) failed or produced no data for plots/tables.")
        
        # --- Linear Moderation (only once per transformation type) ---
        # Typically linear moderation is done on 'raw' or 'zscore', but can be run on others for comparison
        if transformation == "raw" or transformation == "zscore": # Example: run linear only for these two
            logger.info("-" * 60)
            logger.info(f"Linear Moderation on {transformation} data")
            model_linear, df_used_linear = run_linear_moderation(
                df_transformed, iv_t, mo_t, dv_t, final_controls, logger
            )

            if model_linear and df_used_linear is not None and not df_used_linear.empty:
                plot_path_linear = os.path.join(plot_dir, f"linear_{transformation}_{IV3}x{MO1}_{DV2}.png")
                plot_linear_moderation(df_used_linear, iv_t, mo_t, dv_t, model_linear, out_path=plot_path_linear, logger=logger)

                simple_plot_path_linear = os.path.join(simple_plot_dir, f"simple_linear_{transformation}_{IV3}x{MO1}_{DV2}.png")
                plot_simple_moderation(df_used_linear, iv_t, mo_t, dv_t, model_linear,
                                    is_polynomial=False, transform_type=transformation,
                                    out_path=simple_plot_path_linear, logger=logger)
                
                create_research_table_for_linear_path(iv_t, mo_t, dv_t, model_linear, results_dir, logger, transform_type=transformation)
            else:
                logger.warning(f"Linear moderation ({transformation}) failed or produced no data for plots/tables.")

    logger.info("="*80)
    logger.info("All moderation analyses completed.")
    logger.info("="*80)


def plot_linear_moderation(df_plot_data, iv_col, mo_col, dv_col, model, out_path=None, logger=None):
    """
    Creates a visualization of linear moderation effects (scatter + 3D surface).
    df_plot_data: DataFrame containing the data (e.g., df_transformed or df_original).
    iv_col, mo_col, dv_col: Names of columns in df_plot_data.
    """
    if not VISUALIZATION_AVAILABLE:
        if logger: logger.warning("Visualization libraries not found. Skipping plot_linear_moderation.")
        return

    fig = None
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
        iv_label = iv_col.replace('_', ' ').title()
        dv_label = dv_col.replace('_', ' ').title()
        mo_label = mo_col.replace('_', ' ').title()
    
        fig.suptitle(f"Linear Moderation: {iv_label} × {mo_label} → {dv_label}", 
                    fontsize=16, fontweight='bold', y=0.98)
    
        # Moderator levels for lines and point coloring based on mo_col in df_plot_data
        q33 = df_plot_data[mo_col].quantile(0.33)
        q66 = df_plot_data[mo_col].quantile(0.66)
        median_mo = df_plot_data[mo_col].median()

        mo_plot_levels = sorted(list(set([q33, median_mo, q66])))
        if len(mo_plot_levels) == 1: 
            std_mo = df_plot_data[mo_col].std()
            if pd.isna(std_mo) or std_mo == 0 : std_mo = 1 
            mo_plot_levels = [median_mo - std_mo, median_mo, median_mo + std_mo]
        elif len(mo_plot_levels) == 2: 
             mo_plot_levels = [mo_plot_levels[0], (mo_plot_levels[0]+mo_plot_levels[1])/2 , mo_plot_levels[1]]

        mo_low_val_for_line = mo_plot_levels[0]
        mo_med_val_for_line = mo_plot_levels[1] if len(mo_plot_levels) > 1 else mo_plot_levels[0]
        mo_high_val_for_line = mo_plot_levels[-1]

        if logger: 
            logger.info(f"Plotting linear lines for MO levels (from {mo_col}): Low={mo_low_val_for_line:.4f}, Med={mo_med_val_for_line:.4f}, High={mo_high_val_for_line:.4f}")

        df_scatter_plot = df_plot_data.copy()
        conditions_points = [
            df_scatter_plot[mo_col] <= q33,
            (df_scatter_plot[mo_col] > q33) & (df_scatter_plot[mo_col] <= q66),
            df_scatter_plot[mo_col] > q66
        ]
        choices_points = ['Low', 'Medium', 'High']
        df_scatter_plot['mo_group_display'] = np.select(conditions_points, choices_points, default='Unknown')
        
        group_counts = df_scatter_plot['mo_group_display'].value_counts()
        if logger: logger.info(f"Point coloring groups (from {mo_col}): {group_counts.to_dict()}")
        df_scatter_plot['mo_group_display'] = pd.Categorical(df_scatter_plot['mo_group_display'], categories=['Low', 'Medium', 'High', 'Unknown'], ordered=True)
        
        iv_min_plot = df_plot_data[iv_col].min()
        iv_max_plot = df_plot_data[iv_col].max()
        iv_vals_for_lines = np.linspace(iv_min_plot, iv_max_plot, 100)
        
        palette = {"Low": "#d62728", "Medium": "#1f77b4", "High": "#2ca02c", "Unknown": "grey"}
        line_styles = ['-', '--', '-.']
        mo_vals_for_lines_dict = {'Low': mo_low_val_for_line, 'Medium': mo_med_val_for_line, 'High': mo_high_val_for_line}
        
        sns.scatterplot(data=df_scatter_plot, x=iv_col, y=dv_col, hue='mo_group_display', palette=palette,
                      alpha=0.5, s=35, edgecolor='w', linewidth=0.5, ax=ax1)
        
        legend_lines_handles = []
        for i, (group_name, mo_val_iter) in enumerate(mo_vals_for_lines_dict.items()):
            pred_data = calculate_linear_conditional_effects(model, mo_val_iter, iv_vals_for_lines)
            if pred_data is None or pred_data['mean'] is None or len(pred_data['mean']) != len(iv_vals_for_lines):
                if logger: logger.warning(f"Prediction failed or returned unexpected data for MO group '{group_name}' in plot_linear_moderation.")
                continue
            
            line_color = palette.get(group_name, "black")
            mean_pred, lower_band, upper_band = pred_data['mean'], pred_data['ci_lower'], pred_data['ci_upper']

            ax1.fill_between(iv_vals_for_lines, lower_band, upper_band, color=line_color, alpha=0.2)
            
            slope_str = ""
            if len(mean_pred) > 1 and len(iv_vals_for_lines) > 1 and (iv_vals_for_lines[-1] - iv_vals_for_lines[0]) != 0:
                slope = (mean_pred[-1] - mean_pred[0]) / (iv_vals_for_lines[-1] - iv_vals_for_lines[0])
                slope_str = f"Slope: {slope:.3f}"
            
            label = f"{group_name} {mo_label} ({slope_str})"
            line, = ax1.plot(iv_vals_for_lines, mean_pred, color=line_color, 
                          linestyle=line_styles[i % len(line_styles)], linewidth=3, label=label)
            legend_lines_handles.append(line)
        
        ax1.set_xlabel(iv_label, fontsize=12)
        ax1.set_ylabel(dv_label, fontsize=12)
        ax1.set_title('Linear Moderation Effect', fontsize=13)
        if legend_lines_handles:
            try:
                existing_handles, existing_labels = ax1.get_legend_handles_labels()
            except Exception as e:
                if logger: logger.warning(f"Could not get existing legend handles/labels: {e}")
                existing_handles, existing_labels = [], []
            
            # Combine with our line handles and their labels
            combined_handles = existing_handles + legend_lines_handles
            combined_labels = existing_labels + [h.get_label() for h in legend_lines_handles]
            
            # Create new legend with combined items
            ax1.legend(handles=combined_handles, labels=combined_labels, title=f"{mo_label} Levels")
        
        if hasattr(model, 'f_squared'):
            ax1.annotate(
                f"Cohen's f² = {model.f_squared:.4f} ({model.f_squared_interpretation})",
                xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=11, fontweight='bold'
            )
        
        iv_grid_3d = np.linspace(df_plot_data[iv_col].min(), df_plot_data[iv_col].max(), 20)
        mo_grid_3d = np.linspace(df_plot_data[mo_col].min(), df_plot_data[mo_col].max(), 20)
        IV_mesh, MO_mesh = np.meshgrid(iv_grid_3d, mo_grid_3d)
        
        DV_pred_3d = np.full_like(IV_mesh, np.nan)
        for r_idx in range(MO_mesh.shape[0]):
            mo_val_for_surface = MO_mesh[r_idx, 0]
            pred_for_surface_row = calculate_linear_conditional_effects(model, mo_val_for_surface, iv_grid_3d)
            if pred_for_surface_row is not None and len(pred_for_surface_row['mean']) == len(iv_grid_3d):
                 DV_pred_3d[r_idx, :] = pred_for_surface_row['mean']

        if not np.isnan(DV_pred_3d).all():
            surf = ax2.plot_surface(IV_mesh, MO_mesh, DV_pred_3d, cmap='viridis', alpha=0.8, edgecolor='none')
            fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=10, pad=0.1)
        else:
            if logger: logger.warning("Could not generate 3D surface data for linear plot.")

        ax2.set_xlabel(iv_label, fontsize=12, labelpad=10)
        ax2.set_ylabel(mo_label, fontsize=12, labelpad=10)
        ax2.set_zlabel(dv_label, fontsize=12, labelpad=10)
        ax2.view_init(elev=20, azim=-60)
        ax2.set_title('3D Interaction Surface', fontsize=13)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if out_path:
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            if logger: logger.info(f"Saved linear moderation plot to: {out_path}")
        
    except Exception as e:
        if logger: logger.error(f"Error creating plot_linear_moderation: {e}")
        # import traceback
        # if logger: logger.error(traceback.format_exc())
    finally:
        if fig is not None and isinstance(fig, plt.Figure):
            plt.close(fig)


if __name__ == "__main__":
    if VISUALIZATION_AVAILABLE:
        sns.set_context("notebook", font_scale=1.1)
    main()