#!/usr/bin/env python3
"""
This script performs Ordinary Least Squares (OLS) regression analysis on US 
county-level health data for dissertation research. It investigates the 
relationship between spatial clustering patterns of technology adoption 
(LISA cluster types and Moran's I values) and health outcomes (premature death), 
while controlling for demographic and socioeconomic factors.

The script performs two main sets of regressions:
1.  Analyzes the effect of categorical LISA cluster types ('HH', 'HL', 'LH', 'LL')
    on premature death rates, with increasingly comprehensive sets of controls.
2.  Analyzes the effect of the continuous Moran's I value on premature death rates,
    using the same sets of control variables.

Key Features:
-   Connects to a PostgreSQL database to fetch analysis data.
-   Uses statsmodels for OLS regression with standard errors clustered by state.
-   Generates publication-quality visualizations for each regression:
    -   Boxplots with data points for categorical independent variables.
    -   Scatterplots with regression lines for simple regressions.
    -   Partial regression plots (added-variable plots) for multiple regressions.
-   Outputs a consolidated summary table of all regression results to a CSV file.
-   Organizes all outputs (logs, plots, CSV) into a dedicated directory.
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
import statsmodels.api as sm

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
    plt.style.use('seaborn-v0_8-whitegrid')
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib or seaborn not found. Visualizations will be disabled.")

##############################################################################
# LOGGING SETUP
##############################################################################
def setup_logger(log_file_name_prefix="dissertation_regression_analysis_log"):
    """Initializes and configures the logger for the script."""
    logger = logging.getLogger("dissertation_regression_analysis")
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
    return logger

##############################################################################
# DATABASE CONNECTION & DATA FETCH
##############################################################################
def connect_to_database(logger):
    """Establishes and tests a connection to the PostgreSQL database."""
    host = os.getenv("POSTGRES_HOST", 'localhost')
    database = os.getenv("POSTGRES_DB", 'Research_TEST')
    user = os.getenv("POSTGRES_USER", 'postgres')
    password = os.getenv("POSTGRESQL_KEY")

    if password is None:
        logger.error("POSTGRESQL_KEY environment variable not set. Cannot connect to database.")
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
    """Fetches and merges the required data for the regression analyses."""
    sql_query = """
    SELECT
        vcm.county_fips,
        -- Dependent Variable (DV)
        vcv.premature_death_raw_value,     -- DV21

        -- Independent Variables (IVs) from Spatial Analysis
        clc.moran_i AS sp1_moran_i,         -- SP1
        clc.cluster_type AS sp3_cluster_type, -- SP3

        -- Control Variables (CT & IV)
        vcm.population,                    -- CT1
        vcm.census_division,               -- CT2
        vcm.health_behaviors_score,        -- IV3
        vcm.social_economic_factors_score  -- IV4
    FROM
        public.vw_conceptual_model_adjpd AS vcm
    LEFT JOIN 
        public.vw_conceptual_model_variables_adjpd AS vcv
        ON vcm.county_fips = vcv.county_fips
    LEFT JOIN
        public.county_lisa_clusters AS clc
        ON vcm.county_fips = clc.county_fips
    WHERE
        vcm.population IS NOT NULL AND CAST(vcm.population AS NUMERIC) > 0
        AND vcv.premature_death_raw_value IS NOT NULL
        AND clc.moran_i IS NOT NULL
        AND clc.cluster_type IS NOT NULL;
    """
    try:
        df = pd.read_sql_query(sql_query, engine)
        logger.info(f"Data for analysis retrieved: {df.shape[0]} rows, {df.shape[1]} columns.")
        if df.empty:
            logger.error("Fetched DataFrame is empty. Check query and data sources.")
            sys.exit(1)

        logger.info("Renaming columns for clarity...")
        rename_map = {
            'premature_death_raw_value': 'dv21_premature_death',
            'population': 'ct1_population',
            'census_division': 'ct2_census_division',
            'health_behaviors_score': 'iv3_health_behaviors',
            'social_economic_factors_score': 'iv4_social_economic',
        }
        df.rename(columns=rename_map, inplace=True)
        
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            logger.info(f"Initial null values per column:\n{nan_counts[nan_counts > 0]}")
        
        return df
    except Exception as e:
        logger.error(f"Database query or initial data processing failed: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

##############################################################################
# DATA PREPARATION
##############################################################################
def prepare_data_for_regression(df_input, logger):
    """Prepares the DataFrame for regression by creating necessary variables."""
    logger.info("Performing data preparations for regression...")
    df = df_input.copy()
    
    # Ensure county_fips is a string for creating state FIPS
    if 'county_fips' in df.columns:
        df['state_fips_for_clustering'] = df['county_fips'].astype(str).str.zfill(5).str[:2]
        logger.info("Created 'state_fips_for_clustering' column for clustered SEs.")
    else:
        logger.error("'county_fips' column not found. Cannot create state_fips for clustering.")
        return None

    # FIXED: Convert population to numeric first before comparing
    if 'ct1_population' in df.columns:
        # Convert population to numeric type
        logger.info("Converting 'ct1_population' to numeric type...")
        df['ct1_population'] = pd.to_numeric(df['ct1_population'], errors='coerce')
        
        # Now check if all values are positive
        if df['ct1_population'].gt(0).all():
            df['log_population'] = np.log(df['ct1_population'])
            logger.info("Created 'log_population' from 'ct1_population'.")
        else:
            # Handle zero or negative values
            logger.warning("'ct1_population' has zero or negative values. Applying fix for log transform.")
            # Replace zeros and negative values with 1 before log transform
            df['log_population'] = np.log(df['ct1_population'].clip(lower=1))
    else:
        logger.error("'ct1_population' column not found. Cannot create log_population.")
        return None

    # Create dummy variables for census division, dropping one to avoid multicollinearity
    if 'ct2_census_division' in df.columns:
        df['ct2_census_division'] = df['ct2_census_division'].astype(str)
        try:
            # Use dtype=int to ensure dummy variables are 0/1, not True/False
            census_dummies = pd.get_dummies(df['ct2_census_division'], prefix='div', drop_first=True, dtype=int)
            df = pd.concat([df, census_dummies], axis=1)
            logger.info(f"Created {len(census_dummies.columns)} dummy variables for 'ct2_census_division'.")
        except Exception as e:
            logger.error(f"Failed to create dummy variables for census_division: {e}")
    else:
        logger.warning("'ct2_census_division' column not found. Skipping dummy variable creation.")

    # Prepare the categorical IV by setting a reference category
    if 'sp3_cluster_type' in df.columns:
        # Explicitly set the order and make "Not Significant" the reference category
        categories = ["HH", "HL", "LH", "LL", "Not Significant"]
        df['sp3_cluster_type'] = pd.Categorical(df['sp3_cluster_type'], categories=categories, ordered=False)
        logger.info("Prepared 'sp3_cluster_type' as a categorical variable.")

    # Convert all potential regression variables to numeric to be safe
    cols_to_convert = ['dv21_premature_death', 'sp1_moran_i', 'iv3_health_behaviors', 'iv4_social_economic']
    logger.info("Converting all regression variables to proper numeric types...")
    for col in cols_to_convert:
        if col in df.columns:
            # Log sample values to help diagnose issues
            sample_vals = df[col].head(5).tolist()
            logger.info(f"Column '{col}' sample values before conversion: {sample_vals}")
            
            # Convert to numeric, forcing non-convertible values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Log null count after conversion
            null_count = df[col].isnull().sum()
            logger.info(f"Converted '{col}' to numeric type. Null values: {null_count}/{len(df)}")
            
            # Check if any values were coerced to NaN
            if null_count > 0:
                logger.warning(f"'{col}' contains non-numeric values that were converted to NaN.")

    logger.info("Data preparation complete.")
    return df

##############################################################################
# REGRESSION AND VISUALIZATION
##############################################################################

def run_ols_regression(df, dv_name, iv_names, cluster_col, logger):
    """Runs an OLS regression with state-clustered standard errors."""
    try:
        model_cols = [dv_name] + iv_names + [cluster_col]
        model_df = df[model_cols].dropna()

        if model_df.empty or len(model_df) < len(iv_names) + 2:
            logger.warning(f"Insufficient data for model with DV '{dv_name}' after dropping NaNs (N={len(model_df)}).")
            return None, 0

        # Ensure all data is numeric before regression
        Y = pd.to_numeric(model_df[dv_name], errors='coerce')
        
        # Check if dependent variable has any NaN values after conversion
        if Y.isnull().any():
            logger.error(f"DV '{dv_name}' contains NaN values after numeric conversion.")
            return None, 0
            
        # For the X variables, convert each column individually
        X_data = model_df[iv_names].copy()
        for col in iv_names:
            if not pd.api.types.is_numeric_dtype(X_data[col]):
                logger.warning(f"Column '{col}' is not numeric, attempting conversion...")
                X_data[col] = pd.to_numeric(X_data[col], errors='coerce')
        
        # Add constant after ensuring numeric types
        X = sm.add_constant(X_data)
        
        # Log the dtypes to diagnose issues
        logger.info(f"Y dtype: {Y.dtype}")
        logger.info(f"X dtypes: {X.dtypes}")
        
        # Final check on data types before fitting
        if X.dtypes.eq('object').any():
            logger.error(f"Object dtype detected in X variables before fitting: {X.dtypes[X.dtypes.eq('object')]}")
            return None, 0
        
        # Make sure clusters are strings
        clusters = model_df[cluster_col].astype(str)

        model = sm.OLS(Y, X).fit(cov_type='cluster', cov_kwds={'groups': clusters})
        
        logger.info(f"OLS model fitted successfully for DV '{dv_name}' (N={len(model_df)}).")
        return model, len(model_df)

    except Exception as e:
        logger.error(f"OLS regression failed for DV '{dv_name}': {e}")
        logger.debug(traceback.format_exc())
        return None, 0

def plot_categorical_iv(model_df, dv_name, iv_name, controls, model, plot_path):
    """Generates and saves a boxplot for a regression with a categorical IV."""
    if not VISUALIZATION_AVAILABLE: return

    plt.figure(figsize=(12, 8))
    
    # Use stripplot to show individual data points with jitter
    sns.stripplot(data=model_df, x=iv_name, y=dv_name, jitter=0.2, alpha=0.3, color='grey', size=3)
    # Overlay a boxplot to show distribution summaries
    sns.boxplot(data=model_df, x=iv_name, y=dv_name, showfliers=False, boxprops={'facecolor':'None', 'edgecolor':'black'})

    control_str = ", ".join(controls) if controls else "None"
    title = (
        f"Relationship between {iv_name} and {dv_name}\n"
        f"Controls: {control_str}\n"
        f"Model: N={model.nobs}, Adj. R-squared={model.rsquared_adj:.3f}"
    )
    plt.title(title, fontsize=14)
    plt.xlabel(iv_name, fontsize=12)
    plt.ylabel(dv_name, fontsize=12)
    plt.tight_layout()
    
    try:
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Categorical plot saved to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save categorical plot: {e}")

def plot_continuous_iv(dv_name, iv_name, controls, model, plot_path):
    """
    Generates and saves a partial regression plot (added-variable plot) for a 
    continuous IV in a multiple regression context.
    """
    if not VISUALIZATION_AVAILABLE: return

    fig = plt.figure(figsize=(12, 8))
    
    # Check if this is a simple regression or multiple regression
    is_simple_regression = (len(model.model.exog_names) == 2)  # Only 'const' and one IV
    
    try:
        if is_simple_regression:
            # For a simple regression, a standard regplot is most intuitive
            # Reconstruct the data from model endog and exog
            x_data = model.model.exog[:, 1]  # Skip const column
            y_data = model.model.endog
            
            sns.regplot(x=x_data, y=y_data, 
                       line_kws={"color": "red"}, scatter_kws={"alpha": 0.3})
            plt.xlabel(iv_name)
            plt.ylabel(dv_name)
            
            title = (
                f"Simple Regression of {dv_name} on {iv_name}\n"
                f"Coefficient: {model.params[iv_name]:.3f} (p={model.pvalues[iv_name]:.3f}), "
                f"Adj. R-squared={model.rsquared_adj:.3f}"
            )
        else:
            # For multiple regression, use the partial regression plot
            # Reconstruct a DataFrame for statsmodels plot_partregress
            data = pd.DataFrame({
                dv_name: model.model.endog,
                **{name: model.model.exog[:, i] for i, name in enumerate(model.model.exog_names)}
            })
            
            sm.graphics.plot_partregress(
                endog=dv_name,
                exog_i=iv_name,
                exog_others=[c for c in model.model.exog_names if c not in ['const', iv_name]],
                data=data,
                obs_labels=False,
                ax=fig.gca()
            )
            
            control_str = ", ".join(controls) if controls else "None"
            title = (
                f"Partial Regression Plot for {iv_name} on {dv_name}\n"
                f"Controls: {control_str}\n"
                f"Coefficient for {iv_name}: {model.params[iv_name]:.3f} (p={model.pvalues[iv_name]:.3f})"
            )
        
        plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Continuous plot saved to {plot_path}")
        
    except Exception as e:
        logging.error(f"Failed to create regression plot: {e}")
        logging.debug(traceback.format_exc())
        
        # Fallback to simple scatter plot if anything fails
        plt.figure(figsize=(12, 8))
        # Extract data directly from model arrays
        x_data = model.model.exog[:, model.model.exog_names.index(iv_name)]
        y_data = model.model.endog
        
        sns.regplot(x=x_data, y=y_data, 
                  scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        
        plt.title(f"Scatter Plot of {iv_name} vs {dv_name}\n(Fallback - regression plot failed)", 
                fontsize=14)
        plt.xlabel(iv_name)
        plt.ylabel(dv_name)
        plt.tight_layout()
        
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Fallback scatter plot saved to {plot_path}")


##############################################################################
# MAIN SCRIPT EXECUTION
##############################################################################
def main():
    logger = setup_logger()
    logger.info("Starting Dissertation OLS Regression Analysis Script.")
    
    output_dir = "dissertation_regression_output" 
    plot_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    
    warnings.filterwarnings("ignore", category=FutureWarning) 

    # --- Define the regression models to be run ---
    # Control sets
    controls_base = ['log_population', 'census_dummies'] # CT1, CT2
    controls_iv3 = controls_base + ['iv3_health_behaviors']
    controls_iv4 = controls_iv3 + ['iv4_social_economic']

    # List of all regression specifications
    regression_configs = [
        # SP3 (Cluster Type) as IV
        {'name': 'Reg1_SP3_on_DV21', 'dv': 'dv21_premature_death', 'iv': 'sp3_cluster_type', 'controls': []},
        {'name': 'Reg2_SP3_on_DV21_Controls_Base', 'dv': 'dv21_premature_death', 'iv': 'sp3_cluster_type', 'controls': controls_base},
        {'name': 'Reg3_SP3_on_DV21_Controls_IV3', 'dv': 'dv21_premature_death', 'iv': 'sp3_cluster_type', 'controls': controls_iv3},
        {'name': 'Reg4_SP3_on_DV21_Controls_IV4', 'dv': 'dv21_premature_death', 'iv': 'sp3_cluster_type', 'controls': controls_iv4},
        # SP1 (Moran's I) as IV
        {'name': 'Reg5_SP1_on_DV21', 'dv': 'dv21_premature_death', 'iv': 'sp1_moran_i', 'controls': []},
        {'name': 'Reg6_SP1_on_DV21_Controls_Base', 'dv': 'dv21_premature_death', 'iv': 'sp1_moran_i', 'controls': controls_base},
        {'name': 'Reg7_SP1_on_DV21_Controls_IV3', 'dv': 'dv21_premature_death', 'iv': 'sp1_moran_i', 'controls': controls_iv3},
        {'name': 'Reg8_SP1_on_DV21_Controls_IV4', 'dv': 'dv21_premature_death', 'iv': 'sp1_moran_i', 'controls': controls_iv4},
    ]

    try:
        engine = connect_to_database(logger)
        df_raw = fetch_data_for_analysis(engine, logger)
        df_prepared = prepare_data_for_regression(df_raw, logger)
        
        if df_prepared is None:
            sys.exit("Data preparation failed. Exiting.")
            
        # Add this line to calculate and display cluster population summaries
        summarize_clusters_by_population(df_prepared, logger)
        
    except Exception as e_setup:
        logger.error(f"Critical error during data setup: {e_setup}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    all_results = []
    
    # --- Loop through each regression configuration ---
    for config in regression_configs:
        model_name = config['name']
        dv = config['dv']
        iv = config['iv']
        controls = config['controls']
        
        logger.info(f"\n{'='*30} RUNNING REGRESSION: {model_name} {'='*30}")
        logger.info(f"DV: {dv}, IV: {iv}, Controls: {controls}")

        # Prepare list of independent variables for the model
        iv_list_model = []
        is_categorical = (iv == 'sp3_cluster_type')

        if is_categorical:
            # For categorical IV, create dummy variables with int type (0/1)
            # to avoid object dtype issues in statsmodels
            df_model = pd.get_dummies(df_prepared, columns=['sp3_cluster_type'], 
                                      drop_first=False, dtype=int)
            
            # Determine reference category (Not Significant)
            ref_col = 'sp3_cluster_type_Not Significant'
            
            # Only include non-reference category dummy variables
            iv_list_model.extend([c for c in df_model.columns if c.startswith('sp3_cluster_type_') 
                                  and c != ref_col])
            
            logger.info(f"Created dummy variables for '{iv}'. Reference: 'Not Significant'")
            logger.info(f"IV variables for model: {iv_list_model}")
        else:
            df_model = df_prepared.copy()
            iv_list_model.append(iv)
        
        # Add control variables, expanding 'census_dummies' placeholder
        census_dummy_cols = [c for c in df_model.columns if c.startswith('div_')]
        final_controls = []
        for c in controls:
            if c == 'census_dummies':
                final_controls.extend(census_dummy_cols)
            else:
                final_controls.append(c)

        iv_list_model.extend(final_controls)
        
        # Run the regression
        model, n_obs = run_ols_regression(df_model, dv, iv_list_model, 'state_fips_for_clustering', logger)

        if model:
            # Log and store results
            logger.info(f"\n--- Results for {model_name} ---\n{model.summary()}\n")
            
            # Extract key results for summary table
            result_row = {'Model Name': model_name, 'N': n_obs, 'Adj. R-Squared': model.rsquared_adj}
            # Add coefficients and p-values for main IVs
            iv_params = model.params.filter(like=iv)
            iv_pvalues = model.pvalues.filter(like=iv)
            for var, coef in iv_params.items():
                result_row[f'Coef_{var}'] = coef
                result_row[f'P-val_{var}'] = iv_pvalues[var]
            all_results.append(result_row)
            
            # Generate plots
            plot_path = os.path.join(plot_dir, f"{model_name}.png")
            if is_categorical:
                # For plotting, we use the original dataframe before dummy creation
                plot_categorical_iv(df_prepared, dv, iv, final_controls, model, plot_path)
            else: # Continuous IV
                plot_continuous_iv(dv, iv, final_controls, model, plot_path)
        else:
            logger.warning(f"Skipping results and plots for {model_name} due to model failure.")

    # --- Save consolidated results to CSV ---
    if all_results:
        results_df = pd.DataFrame(all_results).set_index('Model Name')
        # Reorder columns for better readability
        cols = results_df.columns.tolist()
        fixed_cols = ['N', 'Adj. R-Squared']
        param_cols = sorted([c for c in cols if c not in fixed_cols])
        results_df = results_df[fixed_cols + param_cols]
        
        results_csv_path = os.path.join(output_dir, "dissertation_regression_summary.csv")
        results_df.to_csv(results_csv_path, float_format='%.4f')
        logger.info(f"\nConsolidated regression results saved to: {results_csv_path}")
        logger.info("\n" + results_df.to_string())
    else:
        logger.info("No regression models were successfully run. No summary table generated.")

    logger.info("\nScript finished. Check the output directory for logs, plots, and summary CSV.")

    # --- Additional Analysis: Summarize Clusters by Population ---
    try:
        logger.info("\n" + "="*30 + " LISA CLUSTER POPULATION SUMMARY " + "="*30)
        
        if 'sp3_cluster_type' not in df_prepared.columns or 'ct1_population' not in df_prepared.columns:
            logger.error("Cannot summarize: missing required columns 'sp3_cluster_type' or 'ct1_population'")
        else:
            # Ensure population is numeric
            df_prepared['ct1_population'] = pd.to_numeric(df_prepared['ct1_population'], errors='coerce')
            
            # Calculate county counts and population by cluster type
            cluster_summary = df_prepared.groupby('sp3_cluster_type').agg(
                county_count=('county_fips', 'count'),
                total_population=('ct1_population', 'sum'),
                avg_population=('ct1_population', 'mean')
            ).reset_index()
            
            # Calculate percentages
            total_counties = df_prepared['county_fips'].nunique()
            total_population = df_prepared['ct1_population'].sum()
            
            cluster_summary['pct_counties'] = (cluster_summary['county_count'] / total_counties) * 100
            cluster_summary['pct_population'] = (cluster_summary['total_population'] / total_population) * 100
            
            # Format for display
            cluster_summary['total_population'] = cluster_summary['total_population'].map(lambda x: f"{int(x):,}")
            cluster_summary['avg_population'] = cluster_summary['avg_population'].map(lambda x: f"{int(x):,}")
            cluster_summary['pct_counties'] = cluster_summary['pct_counties'].map(lambda x: f"{x:.2f}%")
            cluster_summary['pct_population'] = cluster_summary['pct_population'].map(lambda x: f"{x:.2f}%")
            
            # Print the summary table
            logger.info("\nPopulation Distribution by LISA Cluster Type:")
            logger.info("\n" + cluster_summary.to_string(index=False))
            
            # Special focus on HL clusters
            hl_counties = df_prepared[df_prepared['sp3_cluster_type'] == 'HL']
            hl_pop = pd.to_numeric(hl_counties['ct1_population'], errors='coerce').sum()
            logger.info(f"\nHIGHLIGHT - 'HL' Cluster Population: {int(hl_pop):,}")
            logger.info(f"Number of counties in 'HL' cluster: {len(hl_counties)}")
            
            # List the top 10 most populous HL counties
            if len(hl_counties) > 0:
                top_hl = hl_counties.sort_values('ct1_population', ascending=False).head(10)
                logger.info("\nTop 10 most populous counties in 'HL' cluster:")
                for idx, row in top_hl.iterrows():
                    logger.info(f"  • FIPS: {row['county_fips']}, Population: {int(row['ct1_population']):,}")
        
        logger.info("="*80)
        
    except Exception as e_summary:
        logger.error(f"Error in summarizing clusters by population: {e_summary}")
        logger.debug(traceback.format_exc())

def summarize_clusters_by_population(df, logger):
    """
    Summarizes the population distribution across different LISA cluster types.
    """
    logger.info("\n" + "="*30 + " LISA CLUSTER POPULATION SUMMARY " + "="*30)
    
    if 'sp3_cluster_type' not in df.columns or 'ct1_population' not in df.columns:
        logger.error("Cannot summarize: missing required columns 'sp3_cluster_type' or 'ct1_population'")
        return
    
    # Ensure population is numeric
    df['ct1_population'] = pd.to_numeric(df['ct1_population'], errors='coerce')
    
    # Calculate county counts and population by cluster type
    cluster_summary = df.groupby('sp3_cluster_type').agg(
        county_count=('county_fips', 'count'),
        total_population=('ct1_population', 'sum'),
        avg_population=('ct1_population', 'mean')
    ).reset_index()
    
    # Calculate percentages
    total_counties = df['county_fips'].nunique()
    total_population = df['ct1_population'].sum()
    
    cluster_summary['pct_counties'] = (cluster_summary['county_count'] / total_counties) * 100
    cluster_summary['pct_population'] = (cluster_summary['total_population'] / total_population) * 100
    
    # Format for display
    cluster_summary['total_population'] = cluster_summary['total_population'].map(lambda x: f"{int(x):,}")
    cluster_summary['avg_population'] = cluster_summary['avg_population'].map(lambda x: f"{int(x):,}")
    cluster_summary['pct_counties'] = cluster_summary['pct_counties'].map(lambda x: f"{x:.2f}%")
    cluster_summary['pct_population'] = cluster_summary['pct_population'].map(lambda x: f"{x:.2f}%")
    
    # Print the summary table
    logger.info("\nPopulation Distribution by LISA Cluster Type:")
    logger.info("\n" + cluster_summary.to_string(index=False))
    
    # Special focus on HL clusters
    hl_counties = df[df['sp3_cluster_type'] == 'HL']
    hl_pop = pd.to_numeric(hl_counties['ct1_population'], errors='coerce').sum()
    logger.info(f"\nHIGHLIGHT - 'HL' Cluster Population: {int(hl_pop):,}")
    logger.info(f"Number of counties in 'HL' cluster: {len(hl_counties)}")
    
    # List the top 10 most populous HL counties
    if len(hl_counties) > 0:
        top_hl = hl_counties.sort_values('ct1_population', ascending=False).head(10)
        logger.info("\nTop 10 most populous counties in 'HL' cluster:")
        for idx, row in top_hl.iterrows():
            logger.info(f"  • FIPS: {row['county_fips']}, Population: {int(row['ct1_population']):,}")
    
    logger.info("="*80)
    
    return cluster_summary

if __name__ == "__main__":
    main()