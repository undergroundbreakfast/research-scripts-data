#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University  
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
"""
Polynomial Moderation Analysis Script - AI Component Moderation

This script analyzes how individual AI technology components (MO11-MO15) moderate 
the relationship between health behaviors (IV3) and health outcomes (DV15, DV21)
using POLYNOMIAL moderation models to detect nonlinear interaction effects.

Tests both linear and polynomial (degree 2 and 3) moderation effects for:
- IV3 × MO11-15 → DV15 (Health Behaviors × AI Components → Preventable Hospitalizations)
- IV3 × MO11-15 → DV21 (Health Behaviors × AI Components → Premature Death)

Creates visualizations showing:
  - Nonlinear moderation effect curves at different moderator levels
  - 3D interaction surfaces showing how effects change across the full range
  - Confidence bands for all predictions
  - Cohen's f² effect size measures
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
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


##############################################################################
# LOGGING SETUP
##############################################################################

def setup_logger(log_file=None):
    """Simple logger setup."""
    logger = logging.getLogger("ai_component_poly_moderation")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"ai_component_poly_moderation_log_{timestamp}.txt"

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(levelname)s: %(message)s")
    ch.setFormatter(ch_formatter)

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
    """Create a SQLAlchemy engine to connect to Postgres."""
    host = 'localhost'
    database = 'Research_TEST'
    user = 'postgres'
    password = os.getenv("POSTGRESQL_KEY", "YOUR_PASSWORD_HERE")

    if password == "YOUR_PASSWORD_HERE" and os.getenv("POSTGRESQL_KEY") is None:
        logger.warning("Using placeholder password. Update connect_to_database function or set POSTGRESQL_KEY environment variable.")

    try:
        conn_str = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"
        engine = create_engine(conn_str)
        with engine.connect() as cn:
            cn.execute(text("SELECT 1"))
        logger.info("Connected to Postgres successfully.")
        return engine
    except Exception as ex:
        logger.error(f"DB connection failed: {ex}")
        sys.exit(1)


def fetch_combined_data(engine, logger):
    """Fetch data combining IV3, MO11-MO15, DV15, DV21, and controls."""
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
            logger.error("Fetched DataFrame is empty after applying filters.")
            sys.exit(1)

        logger.info(f"Counties: {df['county_fips'].nunique()}")
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
        logger.warning(f"No valid unique values found in {division_col} for dummies.")
        return df, []
    
    valid_divisions = [d for d in unique_divisions if d != 'Unknown']
    if not valid_divisions:
        logger.warning(f"Only 'Unknown' category found in {division_col}.")
        return df, []

    reference_category = valid_divisions[0]
    logger.info(f"Using '{reference_category}' as reference category for {division_col} dummies.")

    dummies = pd.get_dummies(df[division_col], prefix='div', drop_first=False, dtype=int)
    ref_col_name = f'div_{reference_category}'
    dummy_cols_to_keep = [col for col in dummies.columns if col != ref_col_name and col.startswith('div_') and col != 'div_Unknown']

    df_with_dummies = pd.concat([df, dummies[dummy_cols_to_keep]], axis=1)
    
    logger.info(f"Created {len(dummy_cols_to_keep)} dummy cols for {division_col}.")
    return df_with_dummies, dummy_cols_to_keep


##############################################################################
# POLYNOMIAL MODERATION ANALYSIS
##############################################################################

def run_polynomial_moderation(df, iv, mo, dv, controls, logger, degree=2):
    """
    Run polynomial moderation analysis with polynomial terms for IV and IV×MO interactions.
    """
    logger.info(f"Running polynomial moderation (degree {degree}): {dv} ~ {iv}^{degree} * {mo} + controls")
    
    df_model = df.copy()
    required_cols = [iv, mo, dv] + controls
    missing_cols = [col for col in required_cols if col not in df_model.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return None, None
    
    # Check for infinite values
    inf_check = df_model[required_cols].isin([np.inf, -np.inf]).any()
    if inf_check.any():
        logger.warning(f"Infinite values detected, replacing with NaN")
        df_model = df_model.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with missing values
    initial_n = len(df_model)
    df_model = df_model.dropna(subset=required_cols)
    final_n = len(df_model)
    
    logger.info(f"Analysis dataset: {final_n} observations (dropped {initial_n - final_n} rows)")
    
    if final_n < 50:
        logger.warning(f"Small sample size: {final_n} observations")
        if final_n < 20:
            logger.error("Sample size too small for reliable polynomial analysis")
            return None, None
    
    # Center variables
    iv_mean = df_model[iv].mean()
    mo_mean = df_model[mo].mean()
    
    iv_centered = df_model[iv] - iv_mean
    mo_centered = df_model[mo] - mo_mean
    
    # Create polynomial terms for IV
    X = pd.DataFrame(index=df_model.index)
    X['const'] = 1.0
    X[f'{iv}_centered'] = iv_centered
    
    poly_terms = [f'{iv}_centered']
    for p in range(2, degree + 1):
        term_name = f'{iv}_centered^{p}'
        X[term_name] = iv_centered ** p
        poly_terms.append(term_name)
    
    # Add moderator
    X[f'{mo}_centered'] = mo_centered
    
    # Create interaction terms (each polynomial term × moderator)
    interaction_terms = []
    for poly_term in poly_terms:
        interaction_name = f'{poly_term}_x_{mo}_centered'
        X[interaction_name] = X[poly_term] * mo_centered
        interaction_terms.append(interaction_name)
    
    # Add controls
    for control in controls:
        if control not in X.columns:
            X[control] = df_model[control]
    
    y = df_model[dv]
    
    # Check for clean data
    if X.isnull().any().any() or np.isinf(X.values).any():
        logger.error("Design matrix contains NaN or infinite values")
        return None, None
    
    # Fit model
    try:
        model = sm.OLS(y, X).fit()
        
        # Store metadata for predictions and plotting
        model.iv_mean = iv_mean
        model.mo_mean = mo_mean
        model.iv_name = iv
        model.mo_name = mo
        model.dv_name = dv
        model.polynomial_degree = degree
        model.poly_terms = poly_terms
        model.interaction_terms = interaction_terms
        model.controls_in_model = controls
        
        # Calculate Cohen's f²
        r_squared = model.rsquared
        f_squared = r_squared / (1 - r_squared) if r_squared < 1 else np.inf
        model.f_squared = f_squared
        
        if f_squared < 0.02:
            model.f_squared_interpretation = "Negligible effect"
        elif f_squared < 0.15:
            model.f_squared_interpretation = "Small effect"
        elif f_squared < 0.35:
            model.f_squared_interpretation = "Medium effect"
        else:
            model.f_squared_interpretation = "Large effect"
        
        # Log interaction terms significance
        logger.info(f"Model R² = {model.rsquared:.4f}, Cohen's f² = {f_squared:.4f} ({model.f_squared_interpretation})")
        for interaction_term in interaction_terms:
            coef = model.params[interaction_term]
            pval = model.pvalues[interaction_term]
            logger.info(f"  {interaction_term}: β = {coef:.4f}, p = {pval:.4f}")
        
        return model, df_model
    
    except Exception as e:
        logger.error(f"Error fitting polynomial moderation model: {e}")
        return None, None


def calculate_polynomial_conditional_effects(model, mo_value, iv_range):
    """Calculate conditional effects of IV on DV at a specific MO level for polynomial models."""
    try:
        predictions = []
        mo_centered = mo_value - model.mo_mean
        
        for iv_val in iv_range:
            iv_centered = iv_val - model.iv_mean
            
            X_pred = pd.DataFrame()
            X_pred['const'] = [1.0]
            
            # Add polynomial terms
            for poly_term in model.poly_terms:
                if '^' in poly_term:
                    power = int(poly_term.split('^')[-1])
                    X_pred[poly_term] = [iv_centered ** power]
                else:
                    X_pred[poly_term] = [iv_centered]
            
            # Add moderator
            X_pred[f'{model.mo_name}_centered'] = [mo_centered]
            
            # Add interaction terms
            for interaction_term in model.interaction_terms:
                base_poly_term = interaction_term.replace(f'_x_{model.mo_name}_centered', '')
                X_pred[interaction_term] = [X_pred[base_poly_term].iloc[0] * mo_centered]
            
            # Add controls (set to 0)
            for control in model.controls_in_model:
                if control not in X_pred.columns:
                    X_pred[control] = [0.0]
            
            # Order columns to match model
            X_pred = X_pred[model.params.index]
            
            pred_result = model.get_prediction(X_pred)
            summary_frame = pred_result.summary_frame(alpha=0.05)
            
            predictions.append({
                'mean': summary_frame['mean'].iloc[0],
                'ci_lower': summary_frame['mean_ci_lower'].iloc[0],
                'ci_upper': summary_frame['mean_ci_upper'].iloc[0]
            })
        
        return {
            'mean': np.array([p['mean'] for p in predictions]),
            'ci_lower': np.array([p['ci_lower'] for p in predictions]),
            'ci_upper': np.array([p['ci_upper'] for p in predictions])
        }
    
    except Exception as e:
        print(f"Error calculating polynomial conditional effects: {e}")
        return None


##############################################################################
# VISUALIZATION
##############################################################################

def plot_polynomial_moderation(df, iv, mo, dv, model, output_path, logger):
    """Create polynomial moderation plot with 2D curves and 3D surface."""
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available.")
        return
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(16, 7))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        iv_label = iv.replace('_', ' ').title()
        mo_label = mo.replace('_', ' ').title()
        dv_label = dv.replace('_', ' ').title()
        
        fig.suptitle(f"Polynomial Moderation: {iv_label} × {mo_label} → {dv_label} (Degree {model.polynomial_degree})", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Define moderator levels
        q33 = df[mo].quantile(0.33)
        q66 = df[mo].quantile(0.66)
        median_mo = df[mo].median()
        
        mo_levels = {
            'Low': q33,
            'Medium': median_mo,
            'High': q66
        }
        
        # Scatter points colored by MO level
        conditions = [
            df[mo] <= q33,
            (df[mo] > q33) & (df[mo] <= q66),
            df[mo] > q66
        ]
        df_plot = df.copy()
        df_plot['mo_group'] = np.select(conditions, ['Low', 'Medium', 'High'], default='Unknown')
        
        palette = {"Low": "#d62728", "Medium": "#1f77b4", "High": "#2ca02c"}
        sns.scatterplot(data=df_plot, x=iv, y=dv, hue='mo_group', palette=palette,
                       alpha=0.5, s=35, edgecolor='w', linewidth=0.5, ax=ax1)
        
        # Plot conditional effect curves
        iv_range = np.linspace(df[iv].min(), df[iv].max(), 100)
        line_styles = ['-', '--', '-.']
        
        for i, (level_name, mo_val) in enumerate(mo_levels.items()):
            pred_data = calculate_polynomial_conditional_effects(model, mo_val, iv_range)
            
            if pred_data is not None:
                color = palette[level_name]
                
                # Calculate average slope
                slopes = np.gradient(pred_data['mean'], iv_range)
                avg_slope = np.mean(slopes)
                
                ax1.fill_between(iv_range, pred_data['ci_lower'], pred_data['ci_upper'], 
                               color=color, alpha=0.2)
                ax1.plot(iv_range, pred_data['mean'], color=color, 
                        linestyle=line_styles[i], linewidth=3,
                        label=f"{level_name} {mo_label} (Avg slope: {avg_slope:.3f})")
        
        ax1.set_xlabel(iv_label, fontsize=12)
        ax1.set_ylabel(dv_label, fontsize=12)
        ax1.set_title('Nonlinear Moderation Effect', fontsize=13)
        ax1.legend(title=f"{mo_label} Levels")
        
        if hasattr(model, 'f_squared'):
            ax1.annotate(
                f"Cohen's f² = {model.f_squared:.4f} ({model.f_squared_interpretation})",
                xy=(0.5, -0.15), xycoords='axes fraction', ha='center', 
                fontsize=11, fontweight='bold'
            )
        
        # 3D surface
        iv_grid = np.linspace(df[iv].min(), df[iv].max(), 20)
        mo_grid = np.linspace(df[mo].min(), df[mo].max(), 20)
        IV_mesh, MO_mesh = np.meshgrid(iv_grid, mo_grid)
        
        DV_pred = np.full_like(IV_mesh, np.nan)
        for r_idx in range(MO_mesh.shape[0]):
            mo_val = MO_mesh[r_idx, 0]
            pred_row = calculate_polynomial_conditional_effects(model, mo_val, iv_grid)
            if pred_row is not None:
                DV_pred[r_idx, :] = pred_row['mean']
        
        if not np.isnan(DV_pred).all():
            surf = ax2.plot_surface(IV_mesh, MO_mesh, DV_pred, cmap='viridis', 
                                   alpha=0.8, edgecolor='none')
            fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=10, pad=0.1)
        
        ax2.set_xlabel(iv_label, fontsize=12, labelpad=10)
        ax2.set_ylabel(mo_label, fontsize=12, labelpad=10)
        ax2.set_zlabel(dv_label, fontsize=12, labelpad=10)
        ax2.view_init(elev=20, azim=-60)
        ax2.set_title('3D Interaction Surface', fontsize=13)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {output_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating polynomial moderation plot: {e}")


##############################################################################
# RESEARCH TABLE
##############################################################################

def create_research_table(results_list, output_dir, logger):
    """Create research table with polynomial moderation results."""
    try:
        table_data = []
        
        for result in results_list:
            if result.get('model') is not None:
                model = result['model']
                
                # Extract highest-order interaction term
                highest_interaction = result['model'].interaction_terms[-1]
                
                row = {
                    'IV': result['iv'],
                    'Moderator': result['mo'],
                    'DV': result['dv'],
                    'Degree': model.polynomial_degree,
                    'N': int(model.nobs),
                    'R_squared': round(model.rsquared, 4),
                    'Adj_R_squared': round(model.rsquared_adj, 4),
                    'Cohens_f_squared': round(model.f_squared, 4),
                    'Effect_size': model.f_squared_interpretation,
                    'F_statistic': round(model.fvalue, 2),
                    'F_pvalue': round(model.f_pvalue, 4),
                    'Highest_interaction': highest_interaction,
                    'Interaction_coef': round(model.params[highest_interaction], 4),
                    'Interaction_pvalue': round(model.pvalues[highest_interaction], 4),
                    'Significant': 'Yes' if model.pvalues[highest_interaction] < 0.05 else 'No'
                }
                
                table_data.append(row)
        
        if not table_data:
            logger.warning("No successful models for research table")
            return None
        
        results_df = pd.DataFrame(table_data)
        results_df = results_df.sort_values(['DV', 'Degree', 'Moderator'])
        
        output_path = os.path.join(output_dir, 'ai_component_polynomial_moderation_results.csv')
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved research table: {output_path}")
        
        # Log summary
        significant = results_df[results_df['Significant'] == 'Yes']
        logger.info(f"Summary: {len(significant)} of {len(results_df)} interactions significant (p < 0.05)")
        
        if len(significant) > 0:
            logger.info("Significant interactions:")
            for _, row in significant.iterrows():
                logger.info(f"  Degree {row['Degree']}: {row['IV']} × {row['Moderator']} → {row['DV']}: "
                          f"β = {row['Interaction_coef']:.4f}, p = {row['Interaction_pvalue']:.4f}, "
                          f"f² = {row['Cohens_f_squared']:.4f}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error creating research table: {e}")
        return None


##############################################################################
# MAIN
##############################################################################

def main():
    logger = setup_logger()
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")

    output_dir = "ai_component_polynomial_moderation_output"
    plot_dir = os.path.join(output_dir, "plots")
    results_dir = os.path.join(output_dir, "results")
    
    for d in [output_dir, plot_dir, results_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    logger.info(f"Output directory: {output_dir}")

    engine = connect_to_database(logger)
    df_raw = fetch_combined_data(engine, logger)
    
    df_preprocessed, div_dummies = create_census_division_dummies(df_raw, logger)
    
    IV3 = "iv3_health_behaviors"
    DV15 = "dv15_preventable_hospitalizations"
    DV21 = "dv21_premature_death"
    
    moderators = {
        'mo11_ai_staff_scheduling': 'AI Staff Scheduling',
        'mo12_ai_predict_staffing': 'AI Predict Staffing',
        'mo13_ai_predict_demand': 'AI Predict Demand',
        'mo14_ai_routine_tasks': 'AI Routine Tasks',
        'mo15_ai_optimize_workflows': 'AI Optimize Workflows'
    }
    
    outcomes = {DV15: 'Preventable Hospitalizations', DV21: 'Premature Death'}
    controls = ["population"] + div_dummies
    final_controls = [c for c in controls if c in df_preprocessed.columns]
    
    degrees = [2, 3]  # Test polynomial degrees 2 and 3
    results = []
    
    logger.info("="*80)
    logger.info("STARTING POLYNOMIAL MODERATION ANALYSES")
    logger.info("="*80)
    
    for degree in degrees:
        logger.info(f"\n{'='*60}")
        logger.info(f"ANALYZING POLYNOMIAL DEGREE {degree}")
        logger.info(f"{'='*60}")
        
        for mo_var, mo_label in moderators.items():
            for dv_var, dv_label in outcomes.items():
                logger.info("-" * 60)
                logger.info(f"Analyzing: {IV3} × {mo_var} → {dv_var} (Degree {degree})")
                
                model, df_model = run_polynomial_moderation(
                    df_preprocessed, IV3, mo_var, dv_var, final_controls, logger, degree=degree
                )
                
                if model is not None:
                    plot_filename = f"poly_deg{degree}_{IV3}_x_{mo_var}_{dv_var}.png"
                    plot_path = os.path.join(plot_dir, plot_filename)
                    
                    plot_polynomial_moderation(
                        df_model, IV3, mo_var, dv_var, model, plot_path, logger
                    )
                    
                    results.append({
                        'iv': IV3,
                        'mo': mo_var,
                        'dv': dv_var,
                        'model': model,
                        'data': df_model
                    })
                    
                    logger.info(f"✓ Completed: Degree {degree} {mo_label} on {dv_label}")
                else:
                    logger.warning(f"✗ Failed: Degree {degree} {mo_label} on {dv_label}")
    
    logger.info("="*60)
    logger.info("CREATING RESEARCH SUMMARY TABLE")
    logger.info("="*60)
    
    results_table = create_research_table(results, results_dir, logger)
    
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total analyses completed: {len([r for r in results if r['model'] is not None])}")
    logger.info(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    if VISUALIZATION_AVAILABLE:
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style("whitegrid")
    main()
