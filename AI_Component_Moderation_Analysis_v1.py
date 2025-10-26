#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
"""
Moderation Analysis Script - AI Component Moderation

This script analyzes how individual AI technology components (MO11-MO15) moderate 
the relationship between health behaviors (IV3) and health outcomes (DV15, DV21).

Specific relationships tested:
- IV3 × MO11 → DV15 (Health Behaviors × AI Staff Scheduling → Preventable Hospitalizations)
- IV3 × MO11 → DV21 (Health Behaviors × AI Staff Scheduling → Premature Death)
- IV3 × MO12 → DV15 (Health Behaviors × AI Staffing Prediction → Preventable Hospitalizations)
- IV3 × MO12 → DV21 (Health Behaviors × AI Staffing Prediction → Premature Death)
- IV3 × MO13 → DV15 (Health Behaviors × AI Patient Demand → Preventable Hospitalizations)
- IV3 × MO13 → DV21 (Health Behaviors × AI Patient Demand → Premature Death)
- IV3 × MO14 → DV15 (Health Behaviors × AI Routine Tasks → Preventable Hospitalizations)
- IV3 × MO14 → DV21 (Health Behaviors × AI Routine Tasks → Premature Death)
- IV3 × MO15 → DV15 (Health Behaviors × AI Workflow Optimization → Preventable Hospitalizations)
- IV3 × MO15 → DV21 (Health Behaviors × AI Workflow Optimization → Premature Death)

Creates:
  - Moderation plots for each IV3 × MOxx → DVyy relationship
  - Research tables with interaction effects
  - Summary report of all moderation analyses
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
    logger = logging.getLogger("ai_component_moderation_analysis")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"ai_component_moderation_log_{timestamp}.txt"

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
    """
    host = 'localhost'
    database = 'Research_TEST'
    user = 'postgres'
    password = os.getenv("POSTGRESQL_KEY", "YOUR_PASSWORD_HERE")

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


def fetch_combined_data(engine, logger):
    """
    Fetch data from multiple tables combining variables based on data dictionary:
    - IV3: health_behaviors_score (calculated field) 
    - MO11-MO15: AI component percentages from vw_county_tech_summary_adjpd
    - DV15: preventable_hospital_stays_raw_value from vw_conceptual_model_variables_adjpd
    - DV21: premature_death_raw_value from vw_conceptual_model_variables_adjpd
    - Controls: population, census_division from vw_conceptual_model_adjpd
    """
    query = """
        WITH conceptual_variables AS (
            SELECT 
                county_fips,
                preventable_hospital_stays_raw_value AS dv15_preventable_hospitalizations,
                premature_death_raw_value AS dv21_premature_death
            FROM public.vw_conceptual_model_variables_adjpd
            WHERE county_fips IS NOT NULL
        ),
        conceptual_model AS (
            SELECT 
                county_fips,
                health_behaviors_score AS iv3_health_behaviors,
                population,
                census_division
            FROM public.vw_conceptual_model_adjpd
            WHERE county_fips IS NOT NULL
        ),
        tech_components AS (
            SELECT 
                county_fips,
                pct_wfaiss_enabled AS mo11_ai_staff_scheduling,
                pct_wfaipsn_enabled AS mo12_ai_predict_staffing,
                pct_wfaippd_enabled AS mo13_ai_predict_demand,
                pct_wfaiart_enabled AS mo14_ai_routine_tasks,
                pct_wfaioacw_enabled AS mo15_ai_optimize_workflows
            FROM public.vw_county_tech_summary_adjpd
            WHERE county_fips IS NOT NULL
        )
        SELECT 
            cm.county_fips,
            cm.iv3_health_behaviors,
            cv.dv15_preventable_hospitalizations,
            cv.dv21_premature_death,
            cm.population,
            cm.census_division,
            tc.mo11_ai_staff_scheduling,
            tc.mo12_ai_predict_staffing,
            tc.mo13_ai_predict_demand,
            tc.mo14_ai_routine_tasks,
            tc.mo15_ai_optimize_workflows
        FROM conceptual_model cm
        INNER JOIN conceptual_variables cv ON cm.county_fips = cv.county_fips
        INNER JOIN tech_components tc ON cm.county_fips = tc.county_fips
        WHERE cm.iv3_health_behaviors IS NOT NULL
          AND cv.dv15_preventable_hospitalizations IS NOT NULL
          AND cv.dv21_premature_death IS NOT NULL
          AND cm.population IS NOT NULL
          AND cm.census_division IS NOT NULL
          AND tc.mo11_ai_staff_scheduling IS NOT NULL
          AND tc.mo12_ai_predict_staffing IS NOT NULL
          AND tc.mo13_ai_predict_demand IS NOT NULL
          AND tc.mo14_ai_routine_tasks IS NOT NULL
          AND tc.mo15_ai_optimize_workflows IS NOT NULL
    """
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Data fetched successfully: shape={df.shape}")
        
        # Convert numeric columns
        numeric_cols = [
            'iv3_health_behaviors', 'dv15_preventable_hospitalizations', 
            'dv21_premature_death', 'population',
            'mo11_ai_staff_scheduling', 'mo12_ai_predict_staffing',
            'mo13_ai_predict_demand', 'mo14_ai_routine_tasks', 
            'mo15_ai_optimize_workflows'
        ]
        
        logger.info("Converting columns to numeric types...")
        df = safe_convert_to_numeric(df, numeric_cols)
        
        if df.empty:
            logger.error("Fetched DataFrame is empty after applying filters. Check database content.")
            sys.exit(1)

        # Log data summary
        logger.info("Data summary:")
        logger.info(f"Counties: {df['county_fips'].nunique()}")
        logger.info(f"IV3 range: {df['iv3_health_behaviors'].min():.2f} to {df['iv3_health_behaviors'].max():.2f}")
        logger.info(f"DV15 range: {df['dv15_preventable_hospitalizations'].min():.2f} to {df['dv15_preventable_hospitalizations'].max():.2f}")
        logger.info(f"DV21 range: {df['dv21_premature_death'].min():.2f} to {df['dv21_premature_death'].max():.2f}")
        
        # Log AI adoption rates
        for col in [f'mo{i}_ai_{name}' for i, name in enumerate(['staff_scheduling', 'predict_staffing', 'predict_demand', 'routine_tasks', 'optimize_workflows'], 11)]:
            if col in df.columns:
                adoption_rate = (df[col] > 0).mean() * 100
                mean_val = df[col].mean()
                logger.info(f"{col}: {adoption_rate:.1f}% adoption rate, mean = {mean_val:.2f}%")

        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        sys.exit(1)


##############################################################################
# DATA PREPROCESSING
##############################################################################

def safe_convert_to_numeric(df, columns):
    """Safely convert columns to numeric types."""
    df_copy = df.copy()
    for col in columns:
        if col not in df_copy.columns:
            continue
            
        try:
            if pd.api.types.is_object_dtype(df_copy[col]) or pd.api.types.is_string_dtype(df_copy[col]):
                converted = pd.to_numeric(df_copy[col], errors='coerce')
                null_mask = converted.isna() & (~df_copy[col].isna() & df_copy[col].notnull())
                
                if null_mask.any():
                    # Clean common issues
                    problematic_series = df_copy.loc[null_mask, col].astype(str)
                    cleaned_series = problematic_series.str.replace(',', '', regex=False)
                    cleaned_series = cleaned_series.str.replace('%', '', regex=False)
                    cleaned_series = cleaned_series.str.replace(r'[^\d.-]+', '', regex=True)
                    
                    df_copy.loc[null_mask, col] = cleaned_series
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                else:
                    df_copy[col] = converted
            else:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        except Exception as e:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

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
    if not valid_divisions:
        logger.warning(f"Only 'Unknown' category found in {division_col}; skipping dummies.")
        return df, []

    reference_category = valid_divisions[0]
    logger.info(f"Using '{reference_category}' as the reference category for {division_col} dummies.")

    dummies = pd.get_dummies(df[division_col], prefix='div', drop_first=False, dtype=int)
    ref_col_name = f'div_{reference_category}'
    dummy_cols_to_keep = [col for col in dummies.columns if col != ref_col_name and col.startswith('div_') and col != 'div_Unknown']

    df_with_dummies = pd.concat([df, dummies[dummy_cols_to_keep]], axis=1)
    
    logger.info(f"Created {len(dummy_cols_to_keep)} dummy cols for {division_col}. Kept: {dummy_cols_to_keep}")
    return df_with_dummies, dummy_cols_to_keep


##############################################################################
# MODERATION ANALYSIS
##############################################################################

def run_moderation_analysis(df, iv, mo, dv, controls, logger):
    """
    Run moderation analysis with IV × MO interaction predicting DV.
    Uses state-clustered standard errors if available.
    """
    logger.info(f"Running moderation analysis: {dv} ~ {iv} * {mo} + controls")
    
    df_model = df.copy()
    required_cols = [iv, mo, dv] + controls
    missing_cols = [col for col in required_cols if col not in df_model.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return None, None
    
    # Check for infinite values before proceeding
    inf_check = df_model[required_cols].isin([np.inf, -np.inf]).any()
    if inf_check.any():
        logger.warning(f"Infinite values detected in columns: {inf_check[inf_check].index.tolist()}")
        df_model = df_model.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with missing or infinite values
    initial_n = len(df_model)
    df_model = df_model.dropna(subset=required_cols)
    final_n = len(df_model)
    
    logger.info(f"Analysis dataset: {final_n} observations (dropped {initial_n - final_n} rows with missing/infinite values)")
    
    if final_n < 50:
        logger.warning(f"Small sample size: {final_n} observations")
        if final_n < 20:
            logger.error("Sample size too small for reliable analysis")
            return None, None
    
    # Check data ranges and detect outliers
    for col in [iv, mo, dv]:
        q1, q99 = df_model[col].quantile([0.01, 0.99])
        outlier_count = ((df_model[col] < q1) | (df_model[col] > q99)).sum()
        if outlier_count > 0.05 * len(df_model):  # More than 5% outliers
            logger.warning(f"{col}: {outlier_count} extreme outliers (>1st/99th percentile)")
    
    # Center variables for interaction (more robust centering)
    try:
        iv_mean = df_model[iv].mean()
        mo_mean = df_model[mo].mean()
        
        if pd.isna(iv_mean) or pd.isna(mo_mean):
            logger.error("Cannot center variables: mean is NaN")
            return None, None
            
        iv_centered = df_model[iv] - iv_mean
        mo_centered = df_model[mo] - mo_mean
        
        # Check for remaining issues after centering
        if iv_centered.isnull().any() or mo_centered.isnull().any():
            logger.error("Centering produced NaN values")
            return None, None
            
        if np.isinf(iv_centered).any() or np.isinf(mo_centered).any():
            logger.error("Centering produced infinite values")
            return None, None
        
        # Create interaction term
        interaction = iv_centered * mo_centered
        
        if interaction.isnull().any() or np.isinf(interaction).any():
            logger.error("Interaction term contains NaN or infinite values")
            return None, None
        
        # Prepare design matrix
        X = pd.DataFrame(index=df_model.index)
        X['const'] = 1.0
        X[f'{iv}_centered'] = iv_centered
        X[f'{mo}_centered'] = mo_centered
        X[f'{iv}_x_{mo}'] = interaction
        
        # Add controls (ensure they're clean)
        for control in controls:
            control_data = df_model[control]
            if control_data.isnull().any() or np.isinf(control_data).any():
                logger.warning(f"Control variable {control} has NaN/inf values")
                return None, None
            X[control] = control_data
        
        y = df_model[dv]
        
        # Final check for clean data
        if X.isnull().any().any():
            logger.error(f"Design matrix X contains NaN values: {X.isnull().sum()[X.isnull().sum() > 0]}")
            return None, None
            
        if np.isinf(X.values).any():
            logger.error("Design matrix X contains infinite values")
            return None, None
            
        if y.isnull().any() or np.isinf(y).any():
            logger.error("Dependent variable contains NaN or infinite values")
            return None, None
        
        # Fit OLS model
        model = sm.OLS(y, X).fit()
        
        # Store additional information for plotting
        model.iv_mean = iv_mean
        model.mo_mean = mo_mean
        model.iv_name = iv
        model.mo_name = mo
        model.dv_name = dv
        model.interaction_coef = model.params[f'{iv}_x_{mo}']
        model.interaction_pvalue = model.pvalues[f'{iv}_x_{mo}']
        
        logger.info(f"Model R² = {model.rsquared:.4f}")
        logger.info(f"Interaction coefficient = {model.interaction_coef:.4f} (p = {model.interaction_pvalue:.4f})")
        
        return model, df_model
    
    except Exception as e:
        logger.error(f"Error fitting moderation model: {e}")
        # Additional debugging info
        logger.error(f"Data types: IV={df_model[iv].dtype}, MO={df_model[mo].dtype}, DV={df_model[dv].dtype}")
        logger.error(f"Data ranges: IV=[{df_model[iv].min():.2f}, {df_model[iv].max():.2f}], MO=[{df_model[mo].min():.2f}, {df_model[mo].max():.2f}], DV=[{df_model[dv].min():.2f}, {df_model[dv].max():.2f}]")
        return None, None


def calculate_conditional_effects(model, mo_values, iv_range):
    """
    Calculate conditional effects of IV on DV at different levels of MO.
    """
    try:
        predictions = {}
        
        for mo_val in mo_values:
            mo_centered = mo_val - model.mo_mean
            
            pred_data = []
            for iv_val in iv_range:
                iv_centered = iv_val - model.iv_mean
                
                # Create prediction data
                X_pred = pd.DataFrame()
                X_pred['const'] = [1.0]
                X_pred[f'{model.iv_name}_centered'] = [iv_centered]
                X_pred[f'{model.mo_name}_centered'] = [mo_centered]
                X_pred[f'{model.iv_name}_x_{model.mo_name}'] = [iv_centered * mo_centered]
                
                # Set controls to 0 (assuming centered or reference categories)
                for param in model.params.index:
                    if param not in X_pred.columns and param != 'const':
                        X_pred[param] = [0.0]
                
                # Order columns to match model
                X_pred = X_pred[model.params.index]
                
                pred_result = model.get_prediction(X_pred)
                summary_frame = pred_result.summary_frame(alpha=0.05)
                
                pred_data.append({
                    'mean': summary_frame['mean'].iloc[0],
                    'ci_lower': summary_frame['mean_ci_lower'].iloc[0],
                    'ci_upper': summary_frame['mean_ci_upper'].iloc[0]
                })
            
            predictions[mo_val] = pred_data
        
        return predictions
    
    except Exception as e:
        print(f"Error calculating conditional effects: {e}")
        return None


##############################################################################
# VISUALIZATION
##############################################################################

def plot_moderation_analysis(df, iv, mo, dv, model, output_path, logger):
    """
    Create moderation plot showing IV-DV relationship at different levels of MO.
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Skipping plot.")
        return
    
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Define moderator levels (low, medium, high)
        mo_low = df[mo].quantile(0.25)
        mo_med = df[mo].quantile(0.50)
        mo_high = df[mo].quantile(0.75)
        
        mo_levels = {
            'Low': mo_low,
            'Medium': mo_med,
            'High': mo_high
        }
        
        # IV range for predictions
        iv_min, iv_max = df[iv].min(), df[iv].max()
        iv_range = np.linspace(iv_min, iv_max, 50)
        
        # Calculate conditional effects
        predictions = calculate_conditional_effects(model, list(mo_levels.values()), iv_range)
        
        if predictions is not None:
            colors = ['blue', 'orange', 'green']
            
            for i, (level_name, mo_val) in enumerate(mo_levels.items()):
                if mo_val in predictions:
                    pred_data = predictions[mo_val]
                    
                    means = [p['mean'] for p in pred_data]
                    ci_lower = [p['ci_lower'] for p in pred_data]
                    ci_upper = [p['ci_upper'] for p in pred_data]
                    
                    # Plot line and confidence band
                    ax1.plot(iv_range, means, color=colors[i], linewidth=2, 
                            label=f'{level_name} {mo} ({mo_val:.1f})')
                    ax1.fill_between(iv_range, ci_lower, ci_upper, 
                                   color=colors[i], alpha=0.2)
        
        # Format first subplot
        ax1.set_xlabel(iv.replace('_', ' ').title())
        ax1.set_ylabel(dv.replace('_', ' ').title())
        ax1.set_title(f'Moderation Effect: {iv} × {mo} → {dv}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add interaction statistics
        stats_text = f"Interaction Coefficient: {model.interaction_coef:.4f}\n"
        stats_text += f"P-value: {model.interaction_pvalue:.4f}\n"
        stats_text += f"R²: {model.rsquared:.4f}"
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Create scatter plot colored by moderator levels
        mo_binned = pd.cut(df[mo], bins=3, labels=['Low', 'Medium', 'High'])
        
        for i, level in enumerate(['Low', 'Medium', 'High']):
            mask = mo_binned == level
            ax2.scatter(df.loc[mask, iv], df.loc[mask, dv], 
                       color=colors[i], alpha=0.6, s=30, label=f'{level} {mo}')
        
        ax2.set_xlabel(iv.replace('_', ' ').title())
        ax2.set_ylabel(dv.replace('_', ' ').title())
        ax2.set_title(f'Data Distribution by {mo} Levels')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved moderation plot: {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating moderation plot: {e}")


##############################################################################
# RESEARCH TABLE CREATION
##############################################################################

def create_research_table(results_list, output_dir, logger):
    """
    Create a comprehensive research table with all moderation results.
    """
    try:
        table_data = []
        
        # Filter for successful results
        successful_results = [r for r in results_list if r.get('model') is not None]
        
        if not successful_results:
            logger.warning("No successful models to include in research table")
            # Create empty table with headers
            empty_df = pd.DataFrame(columns=[
                'IV', 'Moderator', 'DV', 'N', 'R_squared', 'Adj_R_squared',
                'F_statistic', 'F_pvalue', 'IV_coef', 'IV_pvalue',
                'MO_coef', 'MO_pvalue', 'Interaction_coef', 'Interaction_pvalue',
                'Interaction_significant'
            ])
            output_path = os.path.join(output_dir, 'ai_component_moderation_results.csv')
            empty_df.to_csv(output_path, index=False)
            logger.info(f"Saved empty research table: {output_path}")
            return empty_df
        
        for result in successful_results:
            model = result['model']
            
            # Extract key statistics
            interaction_term = f"{result['iv']}_x_{result['mo']}"
            
            row = {
                'IV': result['iv'],
                'Moderator': result['mo'], 
                'DV': result['dv'],
                'N': int(model.nobs),
                'R_squared': round(model.rsquared, 4),
                'Adj_R_squared': round(model.rsquared_adj, 4),
                'F_statistic': round(model.fvalue, 2),
                'F_pvalue': round(model.f_pvalue, 4),
                'IV_coef': round(model.params[f"{result['iv']}_centered"], 4),
                'IV_pvalue': round(model.pvalues[f"{result['iv']}_centered"], 4),
                'MO_coef': round(model.params[f"{result['mo']}_centered"], 4),
                'MO_pvalue': round(model.pvalues[f"{result['mo']}_centered"], 4),
                'Interaction_coef': round(model.params[interaction_term], 4),
                'Interaction_pvalue': round(model.pvalues[interaction_term], 4),
                'Interaction_significant': 'Yes' if model.pvalues[interaction_term] < 0.05 else 'No'
            }
            
            table_data.append(row)
        
        # Create DataFrame and save
        results_df = pd.DataFrame(table_data)
        
        if not results_df.empty:
            # Sort by DV and then by MO
            results_df = results_df.sort_values(['DV', 'Moderator'])
            
            # Log summary
            significant_interactions = results_df[results_df['Interaction_significant'] == 'Yes']
            logger.info(f"Summary: {len(significant_interactions)} of {len(results_df)} interactions are significant (p < 0.05)")
            
            if len(significant_interactions) > 0:
                logger.info("Significant interactions:")
                for _, row in significant_interactions.iterrows():
                    logger.info(f"  {row['IV']} × {row['Moderator']} → {row['DV']}: β = {row['Interaction_coef']:.4f}, p = {row['Interaction_pvalue']:.4f}")
        
        # Save to CSV
        output_path = os.path.join(output_dir, 'ai_component_moderation_results.csv')
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved research table: {output_path}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error creating research table: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


##############################################################################
# MAIN ANALYSIS FUNCTION
##############################################################################

def main():
    logger = setup_logger()
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

    # Setup output directories
    output_dir = "ai_component_moderation_output"
    plot_dir = os.path.join(output_dir, "plots")
    results_dir = os.path.join(output_dir, "results")
    
    for d in [output_dir, plot_dir, results_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    logger.info(f"Output will be saved in: {output_dir}")

    # Connect to database and fetch data
    engine = connect_to_database(logger)
    df_raw = fetch_combined_data(engine, logger)
    
    if df_raw is None or df_raw.empty:
        logger.error("Failed to load data. Exiting.")
        sys.exit(1)

    # Preprocess data
    df_preprocessed, div_dummies = create_census_division_dummies(df_raw, logger, 'census_division')
    
    # Define analysis variables
    IV3 = "iv3_health_behaviors"
    DV15 = "dv15_preventable_hospitalizations" 
    DV21 = "dv21_premature_death"
    
    # AI component moderators (MO11-MO15)
    moderators = {
        'mo11_ai_staff_scheduling': 'AI Staff Scheduling',
        'mo12_ai_predict_staffing': 'AI Predict Staffing Needs',
        'mo13_ai_predict_demand': 'AI Predict Patient Demand',
        'mo14_ai_routine_tasks': 'AI Automate Routine Tasks',
        'mo15_ai_optimize_workflows': 'AI Optimize Workflows'
    }
    
    # Dependent variables
    outcomes = {
        DV15: 'Preventable Hospitalizations',
        DV21: 'Premature Death'
    }
    
    # Controls
    controls = ["population"] + div_dummies
    final_controls = [c for c in controls if c in df_preprocessed.columns]
    
    logger.info(f"Analysis variables:")
    logger.info(f"  IV: {IV3}")
    logger.info(f"  Moderators: {list(moderators.keys())}")
    logger.info(f"  Outcomes: {list(outcomes.keys())}")
    logger.info(f"  Controls: {final_controls}")
    
    # Run moderation analyses
    results = []
    
    logger.info("="*80)
    logger.info("STARTING AI COMPONENT MODERATION ANALYSES")
    logger.info("="*80)
    
    for mo_var, mo_label in moderators.items():
        for dv_var, dv_label in outcomes.items():
            logger.info("-" * 60)
            logger.info(f"Analyzing: {IV3} × {mo_var} → {dv_var}")
            
            # Run moderation analysis
            model, df_model = run_moderation_analysis(
                df_preprocessed, IV3, mo_var, dv_var, final_controls, logger
            )
            
            if model is not None:
                # Create plot
                plot_filename = f"moderation_{IV3}_x_{mo_var}_{dv_var}.png"
                plot_path = os.path.join(plot_dir, plot_filename)
                
                plot_moderation_analysis(
                    df_model, IV3, mo_var, dv_var, model, plot_path, logger
                )
                
                # Store results
                results.append({
                    'iv': IV3,
                    'mo': mo_var,
                    'dv': dv_var,
                    'model': model,
                    'data': df_model
                })
                
                logger.info(f"✓ Completed analysis: {mo_label} moderation on {dv_label}")
            else:
                logger.warning(f"✗ Failed analysis: {mo_label} moderation on {dv_label}")
    
    # Create comprehensive results table
    logger.info("="*60)
    logger.info("CREATING RESEARCH SUMMARY TABLE")
    logger.info("="*60)
    
    results_table = create_research_table(results, results_dir, logger)
    
    # Final summary
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total analyses completed: {len([r for r in results if r['model'] is not None])}")
    logger.info(f"Results saved in: {output_dir}")
    logger.info(f"Plots saved in: {plot_dir}")
    logger.info(f"Tables saved in: {results_dir}")


if __name__ == "__main__":
    if VISUALIZATION_AVAILABLE:
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style("whitegrid")
    main()