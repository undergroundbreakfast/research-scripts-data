#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University  
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
"""
AI Component 3D Interaction Surface Analysis

Analyzes how AI technology components (MO11-MO15) moderate the relationship between
health behavior components (IV31-IV39) and health outcomes (DV15, DV21).

Creates side-by-side 3D interaction surface plots for each moderator:
- Left panel: DV15 (Preventable Hospitalizations)
- Right panel: DV21 (Premature Death)
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
from scipy import stats

# For plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("ERROR: matplotlib and seaborn required for visualizations")
    sys.exit(1)


##############################################################################
# LOGGING SETUP
##############################################################################

def setup_logger(log_file=None):
    """Simple logger setup."""
    logger = logging.getLogger("ai_component_3d_analysis")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"ai_component_3d_analysis_log_{timestamp}.txt"

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
    """Fetch data with IV31-IV39 (health behavior components), MO11-MO15, and DVs."""
    query = """
        SELECT 
            v.county_fips,
            v.adult_smoking_raw_value AS iv31_adult_smoking,
            v.adult_obesity_raw_value AS iv32_adult_obesity,
            v.food_environment_index_raw_value AS iv33_food_environment,
            v.physical_inactivity_raw_value AS iv34_physical_inactivity,
            v.access_to_exercise_opportunities_raw_value AS iv35_exercise_access,
            v.excessive_drinking_raw_value AS iv36_excessive_drinking,
            v.alcohol_impaired_driving_deaths_raw_value AS iv37_impaired_driving,
            v.sexually_transmitted_infections_raw_value AS iv38_sti_rate,
            v.teen_births_raw_value AS iv39_teen_births,
            v.preventable_hospital_stays_raw_value AS dv15_preventable_hospitalizations,
            v.premature_death_raw_value AS dv21_premature_death,
            m.population,
            m.census_division,
            t.pct_wfaiss_enabled AS mo11_ai_staff_scheduling,
            t.pct_wfaipsn_enabled AS mo12_ai_predict_staffing,
            t.pct_wfaippd_enabled AS mo13_ai_predict_demand,
            t.pct_wfaiart_enabled AS mo14_ai_routine_tasks,
            t.pct_wfaioacw_enabled AS mo15_ai_optimize_workflows
        FROM public.vw_conceptual_model_variables_adjpd v
        INNER JOIN public.vw_conceptual_model_adjpd m 
            ON v.county_fips = m.county_fips
        INNER JOIN public.vw_county_tech_summary_adjpd t 
            ON v.county_fips = t.county_fips
        WHERE v.preventable_hospital_stays_raw_value IS NOT NULL
          AND v.premature_death_raw_value IS NOT NULL
          AND m.population IS NOT NULL
          AND m.census_division IS NOT NULL
          AND v.county_fips IS NOT NULL
    """
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Data fetched successfully: shape={df.shape}")
        
        numeric_cols = [
            'iv31_adult_smoking', 'iv32_adult_obesity', 'iv33_food_environment',
            'iv34_physical_inactivity', 'iv35_exercise_access', 'iv36_excessive_drinking',
            'iv37_impaired_driving', 'iv38_sti_rate', 'iv39_teen_births',
            'dv15_preventable_hospitalizations', 'dv21_premature_death', 'population',
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
# MODERATION ANALYSIS
##############################################################################

def run_moderation_analysis(df, iv, mo, dv, census_dummies, logger):
    """
    Run linear moderation analysis: DV ~ IV + MO + IV*MO + controls
    Returns the fitted model.
    """
    try:
        analysis_df = df[[iv, mo, dv, 'population'] + census_dummies].copy()
        analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan)
        analysis_df = analysis_df.dropna()
        
        if len(analysis_df) < 50:
            logger.warning(f"Insufficient data after dropna: {len(analysis_df)} rows")
            return None
        
        logger.info(f"Analysis dataset: {len(analysis_df)} observations (dropped {len(df) - len(analysis_df)} rows)")
        
        # Center IV and MO
        iv_centered = analysis_df[iv] - analysis_df[iv].mean()
        mo_centered = analysis_df[mo] - analysis_df[mo].mean()
        
        # Create interaction term
        interaction = iv_centered * mo_centered
        
        # Build design matrix
        X = pd.DataFrame({
            'const': 1,
            f'{iv}_centered': iv_centered,
            f'{mo}_centered': mo_centered,
            f'{iv}_x_{mo}': interaction,
            'log_population': np.log(analysis_df['population'] + 1)
        })
        
        # Add census dummies
        for dummy_col in census_dummies:
            if dummy_col in analysis_df.columns:
                X[dummy_col] = analysis_df[dummy_col]
        
        # Check for inf/NaN
        if np.isinf(X.values).any() or np.isnan(X.values).any():
            logger.error("Design matrix contains inf or NaN values")
            return None
        
        y = analysis_df[dv].values
        
        # Fit OLS model
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': analysis_df.index})
        
        # Add metadata
        model.iv_name = iv
        model.mo_name = mo
        model.dv_name = dv
        model.interaction_term = f'{iv}_x_{mo}'
        model.analysis_df = analysis_df
        model.iv_centered = iv_centered
        model.mo_centered = mo_centered
        
        # Calculate Cohen's f²
        r_squared = model.rsquared
        if r_squared < 1.0:
            f_squared = r_squared / (1 - r_squared)
        else:
            f_squared = np.inf
        
        model.f_squared = f_squared
        
        # Interpret effect size
        if f_squared < 0.02:
            interp = "Negligible effect"
        elif f_squared < 0.15:
            interp = "Small effect"
        elif f_squared < 0.35:
            interp = "Medium effect"
        else:
            interp = "Large effect"
        
        model.f_squared_interpretation = interp
        
        logger.info(f"Model R² = {r_squared:.4f}, Cohen's f² = {f_squared:.4f} ({interp})")
        
        interaction_coef = model.params[model.interaction_term]
        interaction_pval = model.pvalues[model.interaction_term]
        logger.info(f"  Interaction: β = {interaction_coef:.4f}, p = {interaction_pval:.4f}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error in moderation analysis: {e}")
        return None


def calculate_conditional_effects(model, mo_value, iv_range):
    """
    Calculate predicted DV at given MO level across IV range.
    Returns dict with 'mean', 'ci_lower', 'ci_upper' arrays.
    """
    try:
        predictions = []
        ci_lowers = []
        ci_uppers = []
        
        for iv_val in iv_range:
            # Center relative to original data means
            iv_centered_val = iv_val - model.analysis_df[model.iv_name].mean()
            mo_centered_val = mo_value - model.analysis_df[model.mo_name].mean()
            interaction_val = iv_centered_val * mo_centered_val
            
            # Create prediction input
            X_pred = pd.DataFrame({
                'const': [1],
                f'{model.iv_name}_centered': [iv_centered_val],
                f'{model.mo_name}_centered': [mo_centered_val],
                f'{model.iv_name}_x_{model.mo_name}': [interaction_val],
                'log_population': [np.log(model.analysis_df['population'].median() + 1)]
            })
            
            # Add census dummies (set to mode or 0)
            for col in model.params.index:
                if col.startswith('div_') and col not in X_pred.columns:
                    X_pred[col] = [0]
            
            # Ensure column order matches model
            X_pred = X_pred[model.params.index]
            
            # Predict
            pred = model.predict(X_pred)
            predictions.append(pred.values[0])
            
            # Calculate CI
            pred_std = np.sqrt(model.scale)
            ci_lower = pred.values[0] - 1.96 * pred_std
            ci_upper = pred.values[0] + 1.96 * pred_std
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
        
        return {
            'mean': np.array(predictions),
            'ci_lower': np.array(ci_lowers),
            'ci_upper': np.array(ci_uppers)
        }
    except Exception as e:
        return None


##############################################################################
# VISUALIZATION - 3D SURFACES ONLY
##############################################################################

def plot_3d_surfaces_sidebyside(df, iv, mo, model_dv15, model_dv21, output_path, logger):
    """
    Create side-by-side 3D interaction surfaces for DV15 and DV21.
    
    Left panel: DV15 (Preventable Hospitalizations)
    Right panel: DV21 (Premature Death)
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available.")
        return
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(18, 8))
        
        iv_label = iv.replace('_', ' ').title()
        mo_label = mo.replace('_', ' ').title()
        
        fig.suptitle(f"3D Interaction Surfaces: {iv_label} × {mo_label}", 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Create meshgrid for surface
        iv_grid = np.linspace(df[iv].min(), df[iv].max(), 25)
        mo_grid = np.linspace(df[mo].min(), df[mo].max(), 25)
        IV_mesh, MO_mesh = np.meshgrid(iv_grid, mo_grid)
        
        # ===== LEFT PANEL: DV15 =====
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        if model_dv15 is not None:
            DV15_pred = np.full_like(IV_mesh, np.nan)
            for r_idx in range(MO_mesh.shape[0]):
                mo_val = MO_mesh[r_idx, 0]
                pred_row = calculate_conditional_effects(model_dv15, mo_val, iv_grid)
                if pred_row is not None:
                    DV15_pred[r_idx, :] = pred_row['mean']
            
            if not np.isnan(DV15_pred).all():
                surf1 = ax1.plot_surface(IV_mesh, MO_mesh, DV15_pred, cmap='coolwarm', 
                                        alpha=0.85, edgecolor='none', linewidth=0)
                fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
            
            # Add interaction stats
            interaction_coef = model_dv15.params[model_dv15.interaction_term]
            interaction_pval = model_dv15.pvalues[model_dv15.interaction_term]
            sig_marker = "***" if interaction_pval < 0.001 else "**" if interaction_pval < 0.01 else "*" if interaction_pval < 0.05 else "ns"
            
            ax1.text2D(0.5, 0.95, 
                      f"β={interaction_coef:.4f} {sig_marker}\nR²={model_dv15.rsquared:.3f}, f²={model_dv15.f_squared:.3f}",
                      transform=ax1.transAxes, fontsize=10, ha='center', 
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax1.set_xlabel(iv_label, fontsize=11, labelpad=10)
        ax1.set_ylabel(mo_label, fontsize=11, labelpad=10)
        ax1.set_zlabel('Preventable Hospitalizations', fontsize=11, labelpad=10)
        ax1.view_init(elev=25, azim=-60)
        ax1.set_title('DV15: Preventable Hospitalizations', fontsize=13, pad=20)
        
        # ===== RIGHT PANEL: DV21 =====
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        if model_dv21 is not None:
            DV21_pred = np.full_like(IV_mesh, np.nan)
            for r_idx in range(MO_mesh.shape[0]):
                mo_val = MO_mesh[r_idx, 0]
                pred_row = calculate_conditional_effects(model_dv21, mo_val, iv_grid)
                if pred_row is not None:
                    DV21_pred[r_idx, :] = pred_row['mean']
            
            if not np.isnan(DV21_pred).all():
                surf2 = ax2.plot_surface(IV_mesh, MO_mesh, DV21_pred, cmap='viridis', 
                                        alpha=0.85, edgecolor='none', linewidth=0)
                fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
            
            # Add interaction stats
            interaction_coef = model_dv21.params[model_dv21.interaction_term]
            interaction_pval = model_dv21.pvalues[model_dv21.interaction_term]
            sig_marker = "***" if interaction_pval < 0.001 else "**" if interaction_pval < 0.01 else "*" if interaction_pval < 0.05 else "ns"
            
            ax2.text2D(0.5, 0.95, 
                      f"β={interaction_coef:.4f} {sig_marker}\nR²={model_dv21.rsquared:.3f}, f²={model_dv21.f_squared:.3f}",
                      transform=ax2.transAxes, fontsize=10, ha='center',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax2.set_xlabel(iv_label, fontsize=11, labelpad=10)
        ax2.set_ylabel(mo_label, fontsize=11, labelpad=10)
        ax2.set_zlabel('Premature Death', fontsize=11, labelpad=10)
        ax2.view_init(elev=25, azim=-60)
        ax2.set_title('DV21: Premature Death', fontsize=13, pad=20)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved 3D surfaces: {output_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating 3D surface plot: {e}")


##############################################################################
# RESEARCH TABLE
##############################################################################

def create_research_table(results_list, output_dir, logger):
    """Create research table with moderation results."""
    try:
        table_data = []
        
        for result in results_list:
            if result.get('model') is not None:
                model = result['model']
                
                interaction_coef = model.params[model.interaction_term]
                interaction_pval = model.pvalues[model.interaction_term]
                
                row = {
                    'IV': result['iv'],
                    'Moderator': result['mo'],
                    'DV': result['dv'],
                    'N': int(model.nobs),
                    'R_squared': round(model.rsquared, 4),
                    'Adj_R_squared': round(model.rsquared_adj, 4),
                    'Cohens_f_squared': round(model.f_squared, 4),
                    'Effect_size': model.f_squared_interpretation,
                    'F_statistic': round(model.fvalue, 2),
                    'F_pvalue': round(model.f_pvalue, 4),
                    'Interaction_term': model.interaction_term,
                    'Interaction_coef': round(interaction_coef, 4),
                    'Interaction_pvalue': round(interaction_pval, 4),
                    'Significant': 'Yes' if interaction_pval < 0.05 else 'No'
                }
                
                table_data.append(row)
        
        if not table_data:
            logger.warning("No successful models for research table")
            return None
        
        results_df = pd.DataFrame(table_data)
        results_df = results_df.sort_values(['IV', 'Moderator', 'DV'])
        
        output_path = os.path.join(output_dir, 'ai_component_3d_analysis_results.csv')
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved research table: {output_path}")
        
        # Log summary
        significant = results_df[results_df['Significant'] == 'Yes']
        logger.info(f"Summary: {len(significant)} of {len(results_df)} interactions significant (p < 0.05)")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error creating research table: {e}")
        return None


##############################################################################
# MAIN ANALYSIS
##############################################################################

def main():
    """Main analysis function."""
    warnings.filterwarnings('ignore')
    
    # Setup
    logger = setup_logger()
    output_dir = "ai_component_3d_analysis_output"
    
    logger.info(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    # Connect and fetch data
    engine = connect_to_database(logger)
    df = fetch_combined_data(engine, logger)
    
    # Create census dummies
    df, census_dummies = create_census_division_dummies(df, logger)
    
    # Define IVs (health behavior components) and moderators (AI components)
    ivs = [
        'iv31_adult_smoking',
        'iv32_adult_obesity',
        'iv33_food_environment',
        'iv34_physical_inactivity',
        'iv35_exercise_access',
        'iv36_excessive_drinking',
        'iv37_impaired_driving',
        'iv38_sti_rate',
        'iv39_teen_births'
    ]
    
    moderators = [
        'mo11_ai_staff_scheduling',
        'mo12_ai_predict_staffing',
        'mo13_ai_predict_demand',
        'mo14_ai_routine_tasks',
        'mo15_ai_optimize_workflows'
    ]
    
    dvs = [
        'dv15_preventable_hospitalizations',
        'dv21_premature_death'
    ]
    
    logger.info("=" * 80)
    logger.info("STARTING 3D INTERACTION SURFACE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"IVs: {len(ivs)} health behavior components (IV31-IV39)")
    logger.info(f"Moderators: {len(moderators)} AI components (MO11-MO15)")
    logger.info(f"DVs: {len(dvs)} health outcomes")
    logger.info(f"Total combinations: {len(ivs)} × {len(moderators)} = {len(ivs) * len(moderators)} (2 DVs per combination)")
    
    results_list = []
    
    # Loop through each IV × Moderator combination
    for iv in ivs:
        for mo in moderators:
            logger.info("-" * 60)
            logger.info(f"Analyzing: {iv} × {mo}")
            
            # Check if IV exists in data
            if iv not in df.columns:
                logger.warning(f"Column {iv} not found in dataframe, skipping")
                continue
            
            # Run models for both DVs
            model_dv15 = None
            model_dv21 = None
            
            for dv in dvs:
                logger.info(f"  Running: {iv} × {mo} → {dv}")
                
                model = run_moderation_analysis(df, iv, mo, dv, census_dummies, logger)
                
                if model is not None:
                    results_list.append({
                        'iv': iv,
                        'mo': mo,
                        'dv': dv,
                        'model': model
                    })
                    
                    if dv == 'dv15_preventable_hospitalizations':
                        model_dv15 = model
                    elif dv == 'dv21_premature_death':
                        model_dv21 = model
            
            # Create 3D surface plot with both DVs side-by-side
            if model_dv15 is not None or model_dv21 is not None:
                output_filename = f"3d_surfaces_{iv}_x_{mo}.png"
                output_path = os.path.join(output_dir, "plots", output_filename)
                
                plot_3d_surfaces_sidebyside(df, iv, mo, model_dv15, model_dv21, output_path, logger)
                logger.info(f"✓ Completed: {iv} × {mo}")
    
    # Create research summary table
    logger.info("=" * 80)
    logger.info("CREATING RESEARCH SUMMARY TABLE")
    logger.info("=" * 80)
    
    results_df = create_research_table(results_list, os.path.join(output_dir, "results"), logger)
    
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total analyses completed: {len(results_list)}")
    logger.info(f"Results saved in: {output_dir}")
    
    if results_df is not None:
        sig_count = len(results_df[results_df['Significant'] == 'Yes'])
        logger.info(f"Significant interactions found: {sig_count} of {len(results_df)}")


if __name__ == "__main__":
    main()
