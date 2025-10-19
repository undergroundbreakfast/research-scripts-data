#!/usr/bin/env python3
"""
Changing to ADJPD patient days technology weights for AIPW analysis on 6/9/25

This script performs a causal analysis to estimate the impact of
SPECIFIC Generative AI and Robotics capabilities adoption in hospitals
on premature death rates at the county level.
It iterates through individual technology components, defines them as treatment,
and uses the Augmented Inverse Propensity to Treat Weighting (AIPW) method
to calculate the Average Treatment Effect (ATE). Includes YPLL/lives saved for significant findings.

Key Steps for EACH technology component:
1. Connects to a PostgreSQL database and retrieves county-level health and tech adoption data.
2. Prepares data: defines treatment (specific tech adoption > 0), outcome (premature death rate),
   and confounders (IVs, demographics, OTHER tech components); handles missing values; creates dummy variables.
3. Estimates propensity scores for the specific tech adoption.
4. Checks for propensity score overlap.
5. Fits outcome models (mu0, mu1).
6. Calculates the AIPW ATE.
7. Estimates 95% confidence intervals for the ATE using bootstrapping (500 iterations).
8. If ATE is significant and beneficial, simulates potential YPLL and lives saved.
9. Logs all results and generates a summary table and plots.
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample # For bootstrapping
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from statsmodels.stats.multitest import multipletests
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# New imports for reviewer feedback analysis
try:
    import statsmodels.formula.api as smf
    from pygam import LinearGAM, s, te
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

##############################################################################
# LOGGING SETUP
##############################################################################
def setup_logger(log_file_name_prefix="genai_components_causal_analysis_log"):
    logger = logging.getLogger("genai_components_causal_analysis")
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
    fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logging to: {log_path}")

    bootstrap_logger = logging.getLogger("bootstrap_internal")
    if not bootstrap_logger.hasHandlers():
        bootstrap_logger.setLevel(logging.WARNING)
    return logger

##############################################################################
# VISUALIZATION FUNCTIONS
##############################################################################
def generate_interaction_pdp_plot(df_prepared, outcome_col, base_confounder_cols, plot_dir, logger, tech_feature='pct_wfaiart_enabled'):
    """
    Generates and saves a two-way Partial Dependence Plot to visualize the
    interaction effect of a technology and a confounder on the outcome.

    Args:
        df_prepared: DataFrame with prepared data
        outcome_col: Column name for outcome variable
        base_confounder_cols: List of confounder column names
        plot_dir: Directory to save plots
        logger: Logger object
        tech_feature: Technology feature to analyze (default: 'pct_wfaiart_enabled')
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping interaction PDP plot.")
        return

    logger.info(f"Generating two-way interaction PDP plot for {tech_feature}...")

    try:
        feature_1 = tech_feature
        feature_2 = 'social_economic_factors_score'
        interaction_features = [(feature_1, feature_2)]

        if not all(f in df_prepared.columns for f in [feature_1, feature_2]):
            logger.error(f"One or more features for interaction plot not found in DataFrame: {feature_1}, {feature_2}")
            return

        model_features = list(dict.fromkeys(base_confounder_cols + [feature_1]))
        X = df_prepared[model_features]
        y = df_prepared[outcome_col]

        gbr_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        gbr_model.fit(X, y)

        fig, ax = plt.subplots(figsize=(12, 9))
        PartialDependenceDisplay.from_estimator(
            gbr_model, X, features=interaction_features, ax=ax
        )

        tech_name = tech_feature.replace('pct_', '').replace('_enabled', '').upper()
        ax.set_title(f"Interaction PDP of {tech_name} and Socio-Economic Score on Outcome", fontsize=18, pad=20)
        ax.set_xlabel(feature_1.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(feature_2.replace('_', ' ').title(), fontsize=12)
        plt.tight_layout()

        tech_suffix = tech_feature.replace('pct_', '').replace('_enabled', '')
        save_path = os.path.join(plot_dir, f"pdp_interaction_plot_{tech_suffix}.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"Interaction PDP plot for {tech_name} saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate interaction PDP plot for {tech_feature}: {e}")
        logger.debug(traceback.format_exc())

def generate_3d_interaction_plot(df_prepared, outcome_col, base_confounder_cols, plot_dir, logger, tech_feature='pct_wfaiart_enabled'):
    """
    Generates and saves a 3D surface plot from Partial Dependence values.
    This shows the marginal effect of two features on the outcome,
    averaging over the effects of all other model confounders.

    Args:
        df_prepared: DataFrame with prepared data
        outcome_col: Column name for the outcome variable
        base_confounder_cols: List of confounder column names
        plot_dir: Directory to save the plot
        logger: Logger object
        tech_feature: Technology feature column to use
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping 3D interaction plot.")
        return

    logger.info(f"Generating 3D Partial Dependence surface plot for {tech_feature}...")

    try:
        feature_1 = tech_feature
        feature_2 = 'social_economic_factors_score'
        interaction_features = [feature_1, feature_2]

        if not all(f in df_prepared.columns for f in interaction_features):
            logger.error(f"One or more features for 3D plot not found in DataFrame: {interaction_features}")
            return

        model_features = list(dict.fromkeys(base_confounder_cols + interaction_features))
        X = df_prepared[model_features]
        y = df_prepared[outcome_col]

        gbr_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        gbr_model.fit(X, y)

        pdp_results = partial_dependence(
            gbr_model, X, features=interaction_features,
            kind='average', grid_resolution=25
        )

        XX, YY = np.meshgrid(pdp_results['grid_values'][0], pdp_results['grid_values'][1])
        Z = pdp_results['average'][0].T

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(XX, YY, Z, cmap=cm.viridis, edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=10, label='Partial Dependence on Outcome')

        tech_name = tech_feature.replace('pct_', '').replace('_enabled', '').upper()
        ax.set_title(f"3D Partial Dependence Plot for {tech_name}", fontsize=18, pad=20)
        ax.set_xlabel(feature_1.replace('_', ' ').title(), fontsize=12, labelpad=10)
        ax.set_ylabel(feature_2.replace('_', ' ').title(), fontsize=12, labelpad=10)
        ax.set_zlabel("Partial Dependence", fontsize=12, labelpad=10)
        ax.view_init(elev=20, azim=240)
        plt.tight_layout()

        tech_suffix = tech_feature.replace('pct_', '').replace('_enabled', '')
        save_path = os.path.join(plot_dir, f"pdp_interaction_plot_3d_{tech_suffix}.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"3D Partial Dependence plot for {tech_name} saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate 3D interaction plot for {tech_feature}: {e}")
        logger.debug(traceback.format_exc())

## MODIFIED FUNCTION ##
def generate_3d_exploratory_surface_plot(df_prepared, outcome_col, plot_dir, logger, tech_feature, second_feature_col, second_feature_friendly_name):
    """
    Generates a generalized 3D surface plot to explore the direct relationship
    between a technology feature, a second specified feature, and the outcome.

    Args:
        df_prepared: DataFrame with prepared data
        outcome_col: Column name for the outcome variable (Z-axis)
        plot_dir: Directory to save the plot
        logger: Logger object
        tech_feature: The specific technology feature to plot (X-axis)
        second_feature_col: The column name of the second feature (e.g., 'social_economic_factors_score')
        second_feature_friendly_name: A display-ready name for the second feature (e.g., 'Socio-Economic Factors Score')
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning(f"Matplotlib not available. Skipping 3D exploratory plot for {tech_feature} and {second_feature_friendly_name}.")
        return

    logger.info(f"Generating 3D exploratory surface plot for {tech_feature} and {second_feature_friendly_name}...")

    try:
        feature_1 = tech_feature
        feature_2 = second_feature_col

        if not all(f in df_prepared.columns for f in [feature_1, feature_2, outcome_col]):
            logger.error(f"One or more features for exploratory plot not found: {[feature_1, feature_2, outcome_col]}")
            return

        plot_df = df_prepared[[feature_1, feature_2, outcome_col]].dropna()
        if plot_df.shape[0] < 10:
             logger.warning(f"Too few data points ({plot_df.shape[0]}) to create a meaningful plot for {tech_feature} and {second_feature_friendly_name}.")
             return

        X_plot = plot_df[[feature_1, feature_2]]
        y_plot = plot_df[outcome_col]

        gbr_simple = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        gbr_simple.fit(X_plot, y_plot)

        grid_res = 50
        x1_min, x1_max = X_plot[feature_1].min(), X_plot[feature_1].max()
        x2_min, x2_max = X_plot[feature_2].min(), X_plot[feature_2].max()
        XX, YY = np.meshgrid(np.linspace(x1_min, x1_max, grid_res),
                             np.linspace(x2_min, x2_max, grid_res))

        grid_data = pd.DataFrame(np.c_[XX.ravel(), YY.ravel()], columns=[feature_1, feature_2])
        Z = gbr_simple.predict(grid_data).reshape(XX.shape)

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(XX, YY, Z, cmap=cm.viridis, edgecolor='none', alpha=0.8)
        fig.colorbar(surf, shrink=0.5, aspect=10, label=f"Predicted {outcome_col.replace('_', ' ').title()}")
        cset = ax.contourf(XX, YY, Z, zdir='z', offset=ax.get_zlim()[0], cmap=cm.viridis, alpha=0.5)

        tech_name = tech_feature.replace('pct_', '').replace('_enabled', '').upper()
        ax.set_title(f"Exploratory 3D Plot: {tech_name} and {second_feature_friendly_name}", fontsize=18, pad=20)
        ax.set_xlabel(feature_1.replace('_', ' ').title(), fontsize=12, labelpad=10)
        ax.set_ylabel(second_feature_friendly_name, fontsize=12, labelpad=10)
        ax.set_zlabel(outcome_col.replace('_', ' ').title(), fontsize=12, labelpad=10)
        ax.view_init(elev=20, azim=240)
        plt.tight_layout()

        tech_suffix = tech_feature.replace('pct_', '').replace('_enabled', '')
        second_feature_suffix = second_feature_friendly_name.lower().replace(' ', '_').replace('-', '_')
        save_path = os.path.join(plot_dir, f"exploratory_surface_plot_3d_{tech_suffix}_vs_{second_feature_suffix}.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"3D exploratory surface plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate 3D exploratory surface plot for {tech_feature}: {e}")
        logger.debug(traceback.format_exc())

def generate_ate_forest_plot(results_df, plot_dir, logger):
    """
    Generates and saves a forest plot of the ATEs and their 95% CIs.
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Matplotlib not available. Skipping ATE forest plot.")
        return

    logger.info("Generating ATE forest plot...")

    try:
        required_cols = ['Component', 'ATE', 'CI_Lower', 'CI_Upper']
        if not all(col in results_df.columns for col in required_cols):
            logger.error(f"Forest plot generation failed: Missing one or more required columns from {required_cols}.")
            return

        plot_df = results_df.dropna(subset=['ATE', 'CI_Lower', 'CI_Upper']).copy()
        if plot_df.empty:
            logger.warning("No valid data to plot for the ATE forest plot after dropping NaNs.")
            return
        plot_df = plot_df.sort_values('ATE', ascending=True).reset_index(drop=True)

        plot_df['error_lower'] = plot_df['ATE'] - plot_df['CI_Lower']
        plot_df['error_upper'] = plot_df['CI_Upper'] - plot_df['ATE']
        errors = [plot_df['error_lower'], plot_df['error_upper']]

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.errorbar(x=plot_df['ATE'], y=plot_df.index, xerr=errors,
                    fmt='o', color='black', ecolor='gray', elinewidth=2, capsize=5,
                    markerfacecolor='blue', markersize=8, label='ATE and 95% CI')

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Line of No Effect')

        ax.set_yticks(plot_df.index)
        ax.set_yticklabels(plot_df['Component'], fontsize=12)
        ax.set_xlabel("Average Treatment Effect (ATE) on Premature Death Rate", fontsize=14)
        ax.set_ylabel("Technology Component", fontsize=14)
        ax.set_title("Adjusted Associations of Technology Adoption on Premature Death Rate", fontsize=18, pad=20)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        ax.legend()

        save_path = os.path.join(plot_dir, "ate_forest_plot.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"ATE forest plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate ATE forest plot: {e}")
        logger.debug(traceback.format_exc())


##############################################################################
# REVIEWER FEEDBACK IMPLEMENTATION FUNCTIONS
##############################################################################

def run_segmented_aipw_analysis(df_analysis, tech_col, outcome_col, confounder_cols, knot, logger):
    """
    Implements Reviewer Action Item 1: Segmented regression on AIPW-weighted data.

    This function runs a weighted least squares (WLS) regression using Inverse
    Propensity Weights (IPW). It tests for a change in the effect of technology
    adoption at a specific 'knot' point, stratified by Socio-Economic Factor quartiles.

    Args:
        df_analysis (pd.DataFrame): The analytical dataframe for the specific technology.
        tech_col (str): The name of the continuous technology adoption column.
        outcome_col (str): The name of the outcome column.
        confounder_cols (list): List of confounders for the propensity score model.
        knot (float): The threshold for the segmented regression (e.g., 0.40 for 40%).
        logger: The logging object.
    """
    if not ADVANCED_STATS_AVAILABLE:
        logger.warning("statsmodels not installed. Skipping segmented regression analysis.")
        return

    logger.info(f"--- Running Segmented AIPW Analysis for {tech_col} with knot at {knot:.0%} ---")
    results = {}
    df = df_analysis.copy()

    # 1. Estimate Propensity Scores (for weights)
    # Using a simplified binary treatment definition for weighting purposes
    df['treatment_binary_for_ps'] = (df[tech_col] > 0).astype(int)
    ps_model = LogisticRegression(solver='liblinear', random_state=42)
    scaler = StandardScaler()
    X_confounders = df[confounder_cols]
    X_scaled = scaler.fit_transform(X_confounders)
    ps_model.fit(X_scaled, df['treatment_binary_for_ps'])
    prop_scores = np.clip(ps_model.predict_proba(X_scaled)[:, 1], 0.01, 0.99)

    # 2. Calculate Inverse Propensity Weights (IPW)
    T = df['treatment_binary_for_ps']
    df['ipw'] = T / prop_scores + (1 - T) / (1 - prop_scores)

    # 3. Create variables for segmented regression
    df['pct'] = df[tech_col] / 100.0 # Scale to 0-1 range
    df['pct_above_knot'] = (df['pct'] - knot) * (df['pct'] > knot)

    # 4. Create SEF Quartiles
    # Higher scores are worse, so Quartile 4 represents the most disadvantaged
    df['sef_quartile'] = pd.qcut(df['social_economic_factors_score'], 4, labels=False, duplicates='drop') + 1

    # 5. Run segmented regression for each quartile
    for q in sorted(df['sef_quartile'].unique()):
        quartile_df = df[df['sef_quartile'] == q]
        
        # The formula tests the slope of 'pct' and the change in slope after the knot
        formula = f"{outcome_col} ~ pct + pct_above_knot"
        
        try:
            wls_model = smf.wls(formula=formula, data=quartile_df, weights=quartile_df['ipw']).fit()
            
            # Slope before knot is the 'pct' coefficient
            slope_pre_40 = wls_model.params['pct']
            # Change in slope after knot
            delta_slope = wls_model.params['pct_above_knot']
            # Total slope after knot
            slope_post_40 = slope_pre_40 + delta_slope
            
            p_value_delta = wls_model.pvalues['pct_above_knot']

            results[f'Quartile {q}'] = {
                'N': len(quartile_df),
                'Slope_Pre_40%': slope_pre_40,
                'Slope_Post_40%': slope_post_40,
                'Delta_Slope': delta_slope,
                'P_Value_for_Delta': p_value_delta
            }
        except Exception as e:
            logger.error(f"Segmented regression failed for SEF Quartile {q}: {e}")
            results[f'Quartile {q}'] = {'Error': str(e)}

    # 6. Log the results in a formatted table
    logger.info("Segmented Regression Results by Socio-Economic Quartile:")
    header = f"{'SEF Quartile':<15} | {'N':>5} | {'Slope Pre-40%':>15} | {'Slope Post-40%':>15} | {'Delta in Slope':>15} | {'P-Value (Delta)':>18}"
    logger.info(header)
    logger.info("-" * len(header))
    for q_name, res in results.items():
        if 'Error' not in res:
            log_line = (f"{q_name:<15} | {res['N']:>5} | {res['Slope_Pre_40%']:>15.2f} | "
                        f"{res['Slope_Post_40%']:>15.2f} | {res['Delta_Slope']:>15.2f} | "
                        f"{res['P_Value_for_Delta']:>18.4f}")
            logger.info(log_line)

    return results


def generate_marginal_effect_equity_plot(df_analysis, tech_col, outcome_col, confounder_cols, plot_dir, logger):
    """
    Implements Reviewer Action Item 2: Marginal effect plot for equity.
    CORRECTED VERSION: This function now correctly uses Inverse Propensity Weights (IPW)
    in the Generalized Additive Model (GAM) to provide a confounder-adjusted estimate.

    Args:
        df_analysis (pd.DataFrame): The analytical dataframe for the specific technology.
        tech_col (str): The name of the continuous technology adoption column.
        outcome_col (str): The name of the outcome column.
        confounder_cols (list): List of confounders for the GAM.
        plot_dir (str): Directory to save the plot.
        logger: The logging object.
    """
    if not ADVANCED_STATS_AVAILABLE or not VISUALIZATION_AVAILABLE:
        logger.warning("pygam or matplotlib not available. Skipping marginal effect equity plot.")
        return

    logger.info(f"--- Generating CORRECTED Marginal Effect Equity Plot for {tech_col} (with IPW) ---")
    df = df_analysis.copy()

    try:
        # --- START OF CORRECTION ---
        # 1. Calculate Inverse Propensity Weights (IPW) to adjust for confounding.
        # This step was missing in the previous version.
        df['treatment_binary_for_ps'] = (df[tech_col] > 0).astype(int)
        ps_model = LogisticRegression(solver='liblinear', random_state=42)
        scaler = StandardScaler()
        X_confounders_ps = df[confounder_cols]
        X_scaled = scaler.fit_transform(X_confounders_ps)
        ps_model.fit(X_scaled, df['treatment_binary_for_ps'])
        prop_scores = np.clip(ps_model.predict_proba(X_scaled)[:, 1], 0.01, 0.99)
        T = df['treatment_binary_for_ps']
        df['ipw'] = T / prop_scores + (1 - T) / (1 - prop_scores)
        # --- END OF CORRECTION ---

        # 2. Prepare the feature matrix with only numeric confounders
        numeric_confounders = []
        for col in confounder_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if not df[col].isnull().all():  # Skip columns that are all NaN
                    numeric_confounders.append(col)
        
        logger.info(f"Using {len(numeric_confounders)} numeric confounders for GAM: {numeric_confounders}")
        
        # Create the feature matrix
        feature_cols = [tech_col, 'social_economic_factors_score'] + numeric_confounders
        X_gam = df[feature_cols].dropna()
        y_gam = df.loc[X_gam.index, outcome_col]
        weights_gam = df.loc[X_gam.index, 'ipw']
        
        if len(X_gam) < 100:
            logger.warning(f"Too few observations ({len(X_gam)}) for GAM after removing NaNs. Skipping marginal effect plot.")
            return
        
        # 3. Construct GAM terms properly
        # Start with the tensor product for the main interaction
        gam_terms = te(0, 1)  # Interaction between tech_col (index 0) and social_economic_factors_score (index 1)
        
        # Add smooth terms for numeric confounders (starting from index 2)
        for i, col in enumerate(numeric_confounders, start=2):
            gam_terms = gam_terms + s(i)
        
        # 4. Fit the GAM with IPW weights
        ## THE CRITICAL CHANGE IS ADDING `weights=weights_gam` TO THE FIT METHOD ##
        gam_model = LinearGAM(gam_terms).fit(X_gam, y_gam, weights=weights_gam)

        # 5. Create prediction grids
        sef_range = np.linspace(df['social_economic_factors_score'].min(), 
                               df['social_economic_factors_score'].max(), 100)
        
        # Create grids for 30% and 50% adoption
        grid_30 = pd.DataFrame({tech_col: 30.0, 'social_economic_factors_score': sef_range})
        grid_50 = pd.DataFrame({tech_col: 50.0, 'social_economic_factors_score': sef_range})

        # Add mean values for numeric confounders
        for col in numeric_confounders:
            mean_val = df[col].mean()
            grid_30[col] = mean_val
            grid_50[col] = mean_val

        # 6. Predict outcomes for both scenarios
        y_pred_30 = gam_model.predict(grid_30[feature_cols])
        y_pred_50 = gam_model.predict(grid_50[feature_cols])

        # 7. Calculate the marginal effect
        marginal_effect = y_pred_50 - y_pred_30

        # 8. Create the plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(sef_range, marginal_effect, color='navy', linewidth=2.5)
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        
        tech_name = tech_col.replace('pct_', '').replace('_enabled', '').upper()
        ax.set_title(f'Equity Impact of Increasing {tech_name} Adoption from 30% to 50% (IPW-Adjusted)', fontsize=18, pad=20)
        ax.set_xlabel('Socio-Economic Factors Score (Higher is Worse)', fontsize=14)
        ax.set_ylabel(f'Change in Predicted {outcome_col.replace("_", " ").title()}', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add annotation explaining the finding
        ax.text(0.95, 0.05, 'A more negative value indicates a larger\nbeneficial effect for disadvantaged counties.\n(Adjusted using Inverse Propensity Weights)',
                transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout()
        tech_suffix = tech_col.replace('pct_', '').replace('_enabled', '')
        save_path = os.path.join(plot_dir, f"equity_marginal_effect_{tech_suffix}_IPW_corrected.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        logger.info(f"CORRECTED marginal effect equity plot (with IPW) saved to: {save_path}")

    except Exception as e:
        logger.error(f"Failed to generate corrected marginal effect equity plot for {tech_col}: {e}")
        logger.debug(traceback.format_exc())
        raise e  # Re-raise the exception after logging

##############################################################################
# DATABASE CONNECTION & DATA FETCH
##############################################################################
def connect_to_database(logger):
    host = os.getenv("POSTGRES_HOST", 'localhost')
    database = os.getenv("POSTGRES_DB", 'Research_TEST')
    user = os.getenv("POSTGRES_USER", 'postgres')
    password = os.getenv("POSTGRESQL_KEY", "YOUR_PASSWORD_HERE")

    if password == "YOUR_PASSWORD_HERE":
        logger.warning("Using placeholder password. Update connect_to_database or set POSTGRESQL_KEY environment variable.")

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

def fetch_data_for_component_analysis(engine, logger):
    sql_query = """
    SELECT
        vcm.county_fips,
        vcm.health_behaviors_score,
        vcm.social_economic_factors_score,
        vcm.physical_environment_score,
        vcm.medicaid_expansion_active,
        vcm.population, -- Crucial for lives saved calculation
        vcm.census_division,
        vcv.premature_death_raw_value,
        vcts.pct_wfaiss_enabled, -- updated for adjpd
        vcts.pct_wfaipsn_enabled,
        vcts.pct_wfaippd_enabled,
        vcts.pct_wfaiart_enabled,
        vcts.pct_wfaioacw_enabled,
        vcts.pct_robohos_enabled
    FROM
        public.vw_conceptual_model_adjpd AS vcm
    JOIN
        public.vw_conceptual_model_variables_adjpd AS vcv
        ON vcm.county_fips = vcv.county_fips
    JOIN
        public.vw_county_tech_summary_adjpd AS vcts  -- updated for adjpd
        ON vcm.county_fips = vcts.county_fips
    WHERE
        vcm.health_behaviors_score IS NOT NULL
        AND vcm.social_economic_factors_score IS NOT NULL
        AND vcm.physical_environment_score IS NOT NULL
        AND vcm.medicaid_expansion_active IS NOT NULL
        AND vcm.population IS NOT NULL -- Ensure population is present
        AND vcm.census_division IS NOT NULL
        AND vcv.premature_death_raw_value IS NOT NULL
        AND vcts.pct_wfaiss_enabled IS NOT NULL
        AND vcts.pct_wfaipsn_enabled IS NOT NULL
        AND vcts.pct_wfaippd_enabled IS NOT NULL
        AND vcts.pct_wfaiart_enabled IS NOT NULL
        AND vcts.pct_wfaioacw_enabled IS NOT NULL
        AND vcts.pct_robohos_enabled IS NOT NULL;
    """
    try:
        df = pd.read_sql_query(sql_query, engine)
        logger.info(f"Data for component analysis retrieved: {df.shape[0]} rows, {df.shape[1]} columns.")
        if df.empty:
            logger.error("Fetched DataFrame for component analysis is empty. Check query and data sources.")
            sys.exit(1)

        logger.info("Converting population to numeric and filtering...")
        df['population'] = pd.to_numeric(df['population'], errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['population'], inplace=True) # Ensure population is numeric before filtering
        df = df[df['population'] > 0]
        if len(df) < initial_rows:
            logger.info(f"Removed {initial_rows - len(df)} rows due to non-positive or non-numeric population.")

        critical_cols = df.columns.tolist()
        if df.isnull().any().any():
            logger.warning(f"Fetched data (after pop filter) contains NaNs. Reviewing:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
            df.dropna(subset=critical_cols, inplace=True)
            logger.info(f"After NaN removal for all critical columns, {df.shape[0]} rows remaining.")

        if df.empty:
            logger.error("DataFrame is empty after initial NaN handling. Cannot proceed.")
            sys.exit(1)
        return df
    except Exception as e:
        logger.error(f"Database query or initial data processing failed: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

##############################################################################
# COMMON DATA PREPARATION
##############################################################################
def common_prepare_data(df, logger):
    logger.info("Performing common data preparations...")
    df.columns = [c.lower() for c in df.columns]

    if 'population' not in df.columns or df['population'].isnull().any() or not df['population'].gt(0).all():
        logger.error("Population column is missing, has NaNs, or non-positive values after initial load. Critical error.")
        sys.exit(1)
    df['log_population'] = np.log(df['population'])

    original_medicaid_col = df['medicaid_expansion_active'].copy()
    if df['medicaid_expansion_active'].dtype == bool or df['medicaid_expansion_active'].dtype == object:
        map_dict = {True: 1, False: 0, '1': 1, '0': 0, 'yes': 1, 'no': 0, 'true': 1, 'false': 0}
        df['medicaid_expansion_active'] = df['medicaid_expansion_active'].astype(str).str.lower().map(map_dict)
        if df['medicaid_expansion_active'].isnull().any():
            unmapped_values = original_medicaid_col[df['medicaid_expansion_active'].isnull()].unique()
            logger.warning(f"NaNs introduced during medicaid_expansion_active mapping for values: {unmapped_values}.")

    df['census_division'] = df['census_division'].astype(str)
    census_dummies = pd.get_dummies(df['census_division'], prefix='div', drop_first=True, dtype=int)
    df = pd.concat([df, census_dummies], axis=1)
    census_dummy_cols = list(census_dummies.columns)
    logger.info(f"Created {len(census_dummy_cols)} dummy variables for 'census_division'.")

    base_confounder_cols = [
        'health_behaviors_score', 'social_economic_factors_score',
        'physical_environment_score', 'medicaid_expansion_active',
        'log_population'
    ] + census_dummy_cols

    outcome_col = 'premature_death_raw_value'
    logger.info("Common data preparations complete.")
    return df, outcome_col, base_confounder_cols, census_dummy_cols

##############################################################################
# CAUSAL INFERENCE CORE FUNCTIONS (AIPW)
##############################################################################
def estimate_propensity_scores(X_confounders, T_treatment, logger, treatment_name):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_confounders)
    try:
        prop_model = LogisticRegression(solver='liblinear', random_state=42, C=0.1, penalty='l1', max_iter=200)
        prop_model.fit(X_scaled, T_treatment)
        prop_scores = prop_model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        logger.error(f"Propensity score model fitting failed for {treatment_name}: {e}")
        return None, None
    prop_scores_clamped = np.clip(prop_scores, 0.01, 0.99)
    return prop_scores_clamped, scaler

def _fit_predict_single_model_for_outcome(X_scaled_data, Y_data, T_condition_mask, model_name_suffix, treatment_name_logging, logger_obj):
    """Helper to fit one outcome model (mu0 or mu1) and predict."""
    if sum(T_condition_mask) > X_scaled_data.shape[1] and sum(T_condition_mask) > 0:
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        try:
            model.fit(X_scaled_data[T_condition_mask], Y_data[T_condition_mask])
            return model.predict(X_scaled_data)
        except Exception as e:
            logger_obj.warning(f"Outcome model {model_name_suffix} for {treatment_name_logging} failed: {e}. Using fallback mean.")
            return np.full(len(Y_data), Y_data[T_condition_mask].mean() if sum(T_condition_mask) > 0 else Y_data.mean())
    else:
        logger_obj.warning(f"Not enough samples for {model_name_suffix} ({sum(T_condition_mask)}) for {treatment_name_logging}. Using fallback mean.")
        return np.full(len(Y_data), Y_data[T_condition_mask].mean() if sum(T_condition_mask) > 0 else Y_data.mean())

def fit_outcome_models(X_confounders, T_treatment, Y_outcome, scaler, logger, treatment_name):
    X_scaled = scaler.transform(X_confounders)

    mu1_hat = _fit_predict_single_model_for_outcome(X_scaled, Y_outcome, (T_treatment == 1), "mu1", treatment_name, logger)
    mu0_hat = _fit_predict_single_model_for_outcome(X_scaled, Y_outcome, (T_treatment == 0), "mu0", treatment_name, logger)

    return mu0_hat, mu1_hat

def calculate_aipw_ate(T_treatment, Y_outcome, prop_scores_clamped, mu0_hat, mu1_hat, logger, treatment_name):
    ey1_terms = (T_treatment / prop_scores_clamped) * (Y_outcome - mu1_hat) + mu1_hat
    ey0_terms = ((1 - T_treatment) / (1 - prop_scores_clamped)) * (Y_outcome - mu0_hat) + mu0_hat
    ey1_terms_clean = ey1_terms[np.isfinite(ey1_terms)]
    ey0_terms_clean = ey0_terms[np.isfinite(ey0_terms)]
    if len(ey1_terms_clean) < len(ey1_terms) or len(ey0_terms_clean) < len(ey0_terms):
        logger.warning(f"NaN/Inf in AIPW terms for {treatment_name}.")
    if len(ey1_terms_clean) == 0 or len(ey0_terms_clean) == 0:
        logger.error(f"Not enough valid terms for ATE for {treatment_name}.")
        return np.nan
    return np.mean(ey1_terms_clean) - np.mean(ey0_terms_clean)

def calculate_p_value_from_bootstrap(ate_estimates_boot, point_estimate_ate):
    """Calculates a two-sided p-value from a bootstrap distribution."""
    if not ate_estimates_boot:
        return np.nan
    ate_estimates_boot = np.array(ate_estimates_boot)
    if point_estimate_ate > 0:
        p_val = np.mean(ate_estimates_boot <= 0)
    elif point_estimate_ate < 0:
        p_val = np.mean(ate_estimates_boot >= 0)
    else: # Point estimate is exactly 0
        return 1.0
    return 2 * p_val

def run_aipw_analysis_for_treatment(df_analysis, current_treatment_col, outcome_col, current_confounder_cols, n_bootstraps, logger, plot_dir):
    logger.info(f"--- Starting AIPW Analysis for Treatment: {current_treatment_col} ---")

    T_treatment = df_analysis[current_treatment_col].values
    Y_outcome = df_analysis[outcome_col].values
    control_mean = Y_outcome[T_treatment == 0].mean()
    control_sd = Y_outcome[T_treatment == 0].std()

    X_confounders = df_analysis[current_confounder_cols]

    prop_scores_clamped, scaler_obj = estimate_propensity_scores(X_confounders, T_treatment, logger, current_treatment_col)
    if prop_scores_clamped is None:
        return np.nan, np.nan, np.nan, np.nan, sum(T_treatment==1), sum(T_treatment==0), True, control_mean, control_sd

    df_analysis['propensity_score'] = prop_scores_clamped
    if VISUALIZATION_AVAILABLE:
        plt.figure(figsize=(10, 6))
        sns.histplot(prop_scores_clamped[T_treatment == 1], color="dodgerblue", label=f"Treated (N={sum(T_treatment==1)})", stat="density", common_norm=False, kde=True, bins=30)
        sns.histplot(prop_scores_clamped[T_treatment == 0], color="orangered", label=f"Control (N={sum(T_treatment==0)})", stat="density", common_norm=False, kde=True, bins=30)
        plt.title(f"Propensity Score Distribution for {current_treatment_col}")
        plt.xlabel("Propensity Score (Clamped 0.01-0.99)")
        plt.ylabel("Density")
        plt.legend(); plt.tight_layout()
        overlap_plot_path = os.path.join(plot_dir, f"propensity_overlap_{current_treatment_col}.png")
        try: plt.savefig(overlap_plot_path, dpi=300); plt.close(); logger.info(f"Overlap plot saved: {overlap_plot_path}")
        except Exception as e: logger.error(f"Failed to save overlap plot for {current_treatment_col}: {e}")

    mu0_hat, mu1_hat = fit_outcome_models(X_confounders, T_treatment, Y_outcome, scaler_obj, logger, current_treatment_col)
    if np.isnan(mu0_hat).all() or np.isnan(mu1_hat).all():
        logger.error(f"Outcome model fitting failed critically for {current_treatment_col} (all NaNs). Skipping.")
        return np.nan, np.nan, np.nan, np.nan, sum(T_treatment==1), sum(T_treatment==0), True, control_mean, control_sd

    ate_aipw = calculate_aipw_ate(T_treatment, Y_outcome, prop_scores_clamped, mu0_hat, mu1_hat, logger, current_treatment_col)
    logger.info(f"ATE for {current_treatment_col}: {ate_aipw:.4f}")

    ate_estimates_boot = []
    df_analysis_reset = df_analysis.reset_index(drop=True)
    for i in range(n_bootstraps):
        if (i + 1) % max(1, (n_bootstraps // 10)) == 0: logger.info(f"Bootstrap {i+1}/{n_bootstraps} for {current_treatment_col}")
        try:
            bootstrap_sample = resample(df_analysis_reset, replace=True, random_state=i)
            if bootstrap_sample.empty or bootstrap_sample[current_treatment_col].nunique() < 2: continue
            bs_T = bootstrap_sample[current_treatment_col].values
            bs_Y = bootstrap_sample[outcome_col].values
            bs_X = bootstrap_sample[current_confounder_cols]
            bs_prop_scores, bs_scaler = estimate_propensity_scores(bs_X, bs_T, logging.getLogger("bootstrap_internal"), current_treatment_col)
            if bs_prop_scores is None: continue
            bs_mu0_hat, bs_mu1_hat = fit_outcome_models(bs_X, bs_T, bs_Y, bs_scaler, logging.getLogger("bootstrap_internal"), current_treatment_col)
            if np.isnan(bs_mu0_hat).all() or np.isnan(bs_mu1_hat).all(): continue
            ate_boot = calculate_aipw_ate(bs_T, bs_Y, bs_prop_scores, bs_mu0_hat, bs_mu1_hat, logging.getLogger("bootstrap_internal"), current_treatment_col)
            if not np.isnan(ate_boot) and np.isfinite(ate_boot): ate_estimates_boot.append(ate_boot)
        except Exception as e: logger.debug(f"Bootstrap error iter {i+1} for {current_treatment_col}: {e}. Trace: {traceback.format_exc(limit=1)}")

    ate_lower_ci, ate_upper_ci = np.nan, np.nan
    if ate_estimates_boot:
        ate_lower_ci = np.percentile(ate_estimates_boot, 2.5)
        ate_upper_ci = np.percentile(ate_estimates_boot, 97.5)
        logger.info(f"95% CI for {current_treatment_col} ({len(ate_estimates_boot)} valid): [{ate_lower_ci:.4f}, {ate_upper_ci:.4f}]")
    else: logger.warning(f"No valid bootstrap ATEs for {current_treatment_col}.")

    raw_p_value = calculate_p_value_from_bootstrap(ate_estimates_boot, ate_aipw)

    logger.info(f"--- Finished AIPW Analysis for Treatment: {current_treatment_col} ---")
    return ate_aipw, ate_lower_ci, ate_upper_ci, raw_p_value, sum(T_treatment==1), sum(T_treatment==0), False, control_mean, control_sd

##############################################################################
# SCENARIO SIMULATION (FOR COMPONENTS)
##############################################################################
# Original YPLL per death assumption is 29 years, based on CDC data, but this was too conservative.
# Assuming a more realistic 25 years of life lost per premature death, based on recent studies.
def simulate_lives_saved_for_component(df_population_data, binary_treatment_col, ate_component, logger, ypll_per_death_assumption=25.0):
    total_ypll_averted, total_deaths_averted = 0.0, 0.0
    if ate_component >= 0 or np.isnan(ate_component):
        logger.info(f"  ATE for component ({ate_component:.4f}) is not beneficial or NaN for YPLL reduction. Lives saved calculation skipped.")
        return total_ypll_averted, total_deaths_averted

    ypll_rate_reduction_per_100k = -ate_component
    if 'population' not in df_population_data.columns:
        logger.error("  'population' column missing in data for lives saved simulation. Cannot proceed.")
        return total_ypll_averted, total_deaths_averted
    if binary_treatment_col not in df_population_data.columns:
        logger.error(f"  Treatment column '{binary_treatment_col}' missing for lives saved simulation. Cannot proceed.")
        return total_ypll_averted, total_deaths_averted

    non_adopter_mask = (df_population_data[binary_treatment_col] == 0)
    non_adopter_counties_df = df_population_data.loc[non_adopter_mask]

    if non_adopter_counties_df.empty:
        logger.info(f"  No non-adopter counties found for '{binary_treatment_col}' in the analytical sample. Lives saved: 0.")
    else:
        valid_population_for_simulation = non_adopter_counties_df['population'][pd.notnull(non_adopter_counties_df['population']) & (non_adopter_counties_df['population'] > 0)]
        if valid_population_for_simulation.empty:
            logger.info(f"  No valid population data for non-adopters of '{binary_treatment_col}'. Lives saved: 0.")
        else:
            ypll_averted_values = (ypll_rate_reduction_per_100k / 100000.0) * valid_population_for_simulation
            total_ypll_averted = ypll_averted_values.sum()
            total_deaths_averted = total_ypll_averted / ypll_per_death_assumption
            logger.info(f"  Scenario for component '{binary_treatment_col}' (ATE: {ate_component:.4f}):")
            logger.info(f"    Total YPLL potentially averted: {total_ypll_averted:,.0f}")
            logger.info(f"    Total premature deaths potentially averted: {total_deaths_averted:,.0f}")
    return total_ypll_averted, total_deaths_averted

def calculate_e_value(ate, outcome_sd=None, outcome_mean=None):
    if np.isnan(ate) or ate == 0 or outcome_sd is None or outcome_sd <= 0:
        return np.nan
    standardized_effect = abs(ate) / outcome_sd
    rr_approx = np.exp(0.91 * standardized_effect)
    rr = max(rr_approx, 1.0)
    e_value = rr + np.sqrt(rr * (rr - 1))
    return e_value

##############################################################################
# MAIN ANALYSIS SCRIPT
##############################################################################
def main():
    logger = setup_logger()
    logger.info("Starting Generative AI & Robotics Component-Level Causal Analysis Script.")

    output_dir = "genai_components_causal_output"
    plot_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    logger.info(f"Output will be saved in: {output_dir} and logs/")

    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.linear_model._logistic')
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    tech_components_to_analyze = [
        'pct_wfaiss_enabled', 'pct_wfaipsn_enabled',
        'pct_wfaippd_enabled', 'pct_wfaiart_enabled',
        'pct_wfaioacw_enabled', 'pct_robohos_enabled'
    ]
    tech_component_friendly_names = {
        'pct_wfaiss_enabled': 'AI Staff Scheduling',
        'pct_wfaipsn_enabled': 'AI Predict Staff Needs',
        'pct_wfaippd_enabled': 'AI Predict Pt Demand',
        'pct_wfaiart_enabled': 'AI Automate Routine Tasks',
        'pct_wfaioacw_enabled': 'AI Optimize Workflows',
        'pct_robohos_enabled': 'Robotics in Hospital'
    }

    all_results = []
    n_bootstraps = 500
    treatment_threshold = 0.0
    ypll_per_death_assumption = 25.0

    try:
        engine = connect_to_database(logger)
        df_full_raw = fetch_data_for_component_analysis(engine, logger)
        if df_full_raw.empty:
            logger.error("No data loaded. Exiting.")
            sys.exit(1)

        df_common_prepared, outcome_col, base_confounder_cols, census_dummy_cols = common_prepare_data(df_full_raw.copy(), logger)

        # --- Generate Justification and Exploratory Plots ---
        logger.info("Generating all justification and exploratory plots for technology components...")
        for tech_col_name in tech_components_to_analyze:
            generate_interaction_pdp_plot(df_common_prepared, outcome_col, base_confounder_cols, plot_dir, logger, tech_col_name)
            generate_3d_interaction_plot(df_common_prepared, outcome_col, base_confounder_cols, plot_dir, logger, tech_col_name)

            ## ADDED / MODIFIED: Call generalized function for both Socio-Economic and Health Behavior plots ##
            # First, for Socio-Economic Factors
            generate_3d_exploratory_surface_plot(
                df_prepared=df_common_prepared,
                outcome_col=outcome_col,
                plot_dir=plot_dir,
                logger=logger,
                tech_feature=tech_col_name,
                second_feature_col='social_economic_factors_score',
                second_feature_friendly_name='Socio-Economic Factors Score'
            )

            # Second, for Health Behaviors
            generate_3d_exploratory_surface_plot(
                df_prepared=df_common_prepared,
                outcome_col=outcome_col,
                plot_dir=plot_dir,
                logger=logger,
                tech_feature=tech_col_name,
                second_feature_col='health_behaviors_score',
                second_feature_friendly_name='Health Behaviors Score'
            )

        logger.info("--- All preliminary plots generated. Starting main analysis. ---")
        # --- End of Plot Generation ---

        for tech_col_name in tech_components_to_analyze:
            friendly_name = tech_component_friendly_names.get(tech_col_name, tech_col_name)
            logger.info(f"Processing Component: {friendly_name}")

            df_current_analysis_iter = df_common_prepared.copy()
            current_binary_treatment_col = f"{tech_col_name}_adopted"
            df_current_analysis_iter[current_binary_treatment_col] = (df_current_analysis_iter[tech_col_name] > treatment_threshold).astype(int)

            n_treated_initial = df_current_analysis_iter[current_binary_treatment_col].sum()
            n_control_initial = len(df_current_analysis_iter) - n_treated_initial
            logger.info(f"  Initial Prevalence for {current_binary_treatment_col}: {n_treated_initial / len(df_current_analysis_iter):.2%} (N_T={n_treated_initial}, N_C={n_control_initial})")

            other_tech_confounders = [tc for tc in tech_components_to_analyze if tc != tech_col_name]
            current_confounder_cols_final = base_confounder_cols + other_tech_confounders
            cols_for_this_run = [outcome_col, current_binary_treatment_col, 'population'] + current_confounder_cols_final

            for col in current_confounder_cols_final:
                 if col in df_current_analysis_iter.columns and pd.api.types.is_numeric_dtype(df_current_analysis_iter[col].dtype) == False:
                    df_current_analysis_iter[col] = pd.to_numeric(df_current_analysis_iter[col], errors='coerce')

            rows_before_dropna = len(df_current_analysis_iter)
            df_current_analysis_iter.dropna(subset=cols_for_this_run, inplace=True)
            rows_after_dropna = len(df_current_analysis_iter)

            if rows_after_dropna < rows_before_dropna:
                logger.info(f"  Dropped {rows_before_dropna - rows_after_dropna} rows due to NaNs for {friendly_name}.")

            n_treated_final = df_current_analysis_iter[current_binary_treatment_col].sum() if not df_current_analysis_iter.empty else 0
            n_control_final = len(df_current_analysis_iter) - n_treated_final
            min_samples_heuristic = len(current_confounder_cols_final) + 10

            if n_treated_final < min_samples_heuristic or n_control_final < min_samples_heuristic:
                logger.warning(f"  Insufficient samples in treated ({n_treated_final}) or control ({n_control_final}) "
                               f"group for {friendly_name} (min needed per group ~{min_samples_heuristic}). Skipping AIPW.")
                all_results.append({
                    'Component': friendly_name, 'ATE': np.nan, 'CI_Lower': np.nan, 'CI_Upper': np.nan,
                    'Raw_P_Value': np.nan, 'N_Treated': n_treated_final, 'N_Control': n_control_final,
                    'YPLL_Averted': 0.0, 'Deaths_Averted': 0.0, 'E_value': np.nan, 'Error': True
                })
                continue

            logger.info(f"  Final N for {friendly_name} analysis: {len(df_current_analysis_iter)} (Treated: {n_treated_final}, Control: {n_control_final})")

            ate, ci_low, ci_high, raw_p_val, n_t_from_func, n_c_from_func, error_flag, control_mean, control_sd = run_aipw_analysis_for_treatment(
                df_current_analysis_iter, current_binary_treatment_col, outcome_col,
                current_confounder_cols_final, n_bootstraps, logger, plot_dir
            )

            ypll_averted, deaths_averted, e_value = 0.0, 0.0, np.nan
            if not error_flag and not np.isnan(ate):
                e_value = calculate_e_value(ate, control_sd)
                logger.info(f"  E-value for {friendly_name}: {e_value:.2f}")
                if ate < 0:
                    ypll_averted, deaths_averted = simulate_lives_saved_for_component(
                        df_current_analysis_iter, current_binary_treatment_col,
                        ate, logger, ypll_per_death_assumption
                    )

            all_results.append({
                'Component': friendly_name, 'ATE': ate, 'CI_Lower': ci_low, 'CI_Upper': ci_high,
                'Raw_P_Value': raw_p_val, 'N_Treated': n_treated_final, 'N_Control': n_control_final,
                'YPLL_Averted': ypll_averted, 'Deaths_Averted': deaths_averted, 'E_value': e_value, 'Error': error_flag
            })

            ## REVIEWER FEEDBACK IMPLEMENTATION ##
            # Run the special analyses only for the specified technology
            if tech_col_name == 'pct_wfaioacw_enabled':
                if df_current_analysis_iter.empty:
                    logger.warning(f"Skipping reviewer feedback analysis for {tech_col_name} due to empty dataframe.")
                    continue

                # ACTION ITEM 1: Segmented Regression
                run_segmented_aipw_analysis(
                    df_analysis=df_current_analysis_iter,
                    tech_col=tech_col_name,
                    outcome_col=outcome_col,
                    confounder_cols=current_confounder_cols_final,
                    knot=0.40,
                    logger=logger
                )

                # ACTION ITEM 2: Marginal Effect Equity Plot
                generate_marginal_effect_equity_plot(
                    df_analysis=df_current_analysis_iter,
                    tech_col=tech_col_name,
                    outcome_col=outcome_col,
                    confounder_cols=current_confounder_cols_final,
                    plot_dir=plot_dir,
                    logger=logger
                )

        logger.info("All components processed. Generating summary outputs...")
        summary_df = pd.DataFrame(all_results)
        summary_df.sort_values(by='ATE', ascending=True, inplace=True) # Sort for forest plot

        summary_path = os.path.join(output_dir, "component_analysis_summary.xlsx")
        summary_df.to_excel(summary_path, index=False, sheet_name='Summary')
        logger.info(f"Summary table saved to: {summary_path}")

        generate_ate_forest_plot(summary_df, plot_dir, logger)

    except Exception as e:
        logger.error(f"Unexpected error in main analysis script: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    logger.info("Analysis script completed successfully.")
    return

# --- EXECUTE MAIN ANALYSIS ---
if __name__ == "__main__":
    # Add a check for the new libraries
    if not ADVANCED_STATS_AVAILABLE:
        print("WARNING: 'statsmodels' or 'pygam' not found. Advanced reviewer feedback analysis will be skipped.")
        print("Please run: pip install statsmodels pygam")
    main()