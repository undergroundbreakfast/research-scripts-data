#!/usr/bin/env python3
"""
This script performs a descriptive and comparative analysis of US counties
based on hospital AI and robotics capabilities. It summarizes key public health
and healthcare metrics for six predefined county categories.

Outputs:
- A summary table (CSV and logged) with descriptive statistics for each metric
  by county category.
- Statistical test results (ANOVA or Kruskal-Wallis) indicating if metrics
  differ significantly across categories.
- Box plots visualizing the distribution of each metric across categories.
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
import statsmodels.api as sm # Primarily for OLS if needed, not main focus here
from scipy.stats import shapiro, levene, f_oneway, kruskal
import matplotlib.colors as mcolors

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
def setup_logger(log_file_name_prefix="county_category_summary_analysis_log"):
    logger = logging.getLogger("county_category_summary_analysis")
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
# DATABASE CONNECTION
##############################################################################
def connect_to_database(logger):
    host = os.getenv("POSTGRES_HOST", 'localhost')
    database = os.getenv("POSTGRES_DB", 'Research_TEST') # Your DB name
    user = os.getenv("POSTGRES_USER", 'postgres')       # Your DB user
    password = os.getenv("POSTGRESQL_KEY", "YOUR_PASSWORD_HERE") # Your DB password

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

##############################################################################
# DATA FETCHING
##############################################################################
def fetch_data_for_summary_analysis(engine, logger):
    sql_query = """
    WITH county_beds AS (
        SELECT
            fcounty AS county_fips, -- fcounty from aha_survey_data is 5-digit FIPS
            SUM(CAST(bdtot AS NUMERIC)) AS total_beds_in_county
        FROM aha_survey_data
        WHERE bdtot IS NOT NULL 
          AND bdtot != '' 
          AND bdtot !~ '[^0-9.]'  -- Only numeric characters and decimal points
          AND CAST(bdtot AS NUMERIC) > 0 -- Ensure beds data is valid
        GROUP BY fcounty
    )
    SELECT
        vcm.county_fips,
        vcm.county_category,
        -- Rurality
        CAST(cac3.rural_raw_value AS NUMERIC) AS rural_raw_value,
        -- Total Beds (per county)
        cb.total_beds_in_county,
        -- PCP Availability
        CAST(vcv.ratio_of_population_to_primary_care_physicians AS NUMERIC) AS ratio_of_population_to_primary_care_physicians,
        -- Life expectancy
        CAST(cac2.life_expectancy_raw_value AS NUMERIC) AS life_expectancy_raw_value,
        -- Premature Deaths (YPLL)
        CAST(vcv.premature_death_raw_value AS NUMERIC) AS dv21_premature_death_ypll_rate,
        -- Child Mortality
        CAST(cac2.child_mortality_raw_value AS NUMERIC) AS child_mortality_raw_value,
        -- Infant Mortality
        CAST(cac2.infant_mortality_raw_value AS NUMERIC) AS infant_mortality_raw_value,
        -- Preventable hospitalizations
        CAST(vcv.preventable_hospital_stays_raw_value AS NUMERIC) AS dv15_preventable_stays_rate,
        -- MO1 score
        CAST(vcm.weighted_ai_adoption_score AS NUMERIC) AS mo1_ai_adoption_score,
        -- MO2 score
        CAST(vcm.weighted_robotics_adoption_score AS NUMERIC) AS mo2_robotics_adoption_score,
        -- Profitability: Patient Services Margin
        CAST(vcm.avg_patient_services_margin AS NUMERIC) AS avg_patient_services_margin,
        -- Population for context
        CAST(vcm.population AS NUMERIC) AS population
    FROM
        public.vw_conceptual_model AS vcm
    LEFT JOIN
        public.chr_analytic_chunk_3 AS cac3 ON vcm.county_fips = cac3._5_digit_fips
    LEFT JOIN
        county_beds AS cb ON vcm.county_fips = cb.county_fips
    LEFT JOIN
        public.vw_conceptual_model_variables AS vcv ON vcm.county_fips = vcv.county_fips
    LEFT JOIN
        public.chr_analytic_chunk_2 AS cac2 ON vcm.county_fips = cac2._5_digit_fips
    WHERE
        vcm.county_category IS NOT NULL;
    """
    try:
        df = pd.read_sql_query(sql_query, engine)
        logger.info(f"Data for summary analysis retrieved: {df.shape[0]} rows, {df.shape[1]} columns.")
        if df.empty:
            logger.error("Fetched DataFrame is empty. Check query, FIPS joins, and data sources (vw_conceptual_model, chr_analytic_chunk_2/3, vw_conceptual_model_variables, aha_survey_data).")
            sys.exit(1)
        
        # Standardize column names to lower case for easier handling
        df.columns = [c.lower() for c in df.columns]
        logger.info("Column names standardized to lower case.")
        
        # Log NaN counts for key columns
        nan_counts = df.isnull().sum()
        logger.info(f"Initial NaN counts per column:\n{nan_counts[nan_counts > 0]}")

        return df
    except Exception as e:
        logger.error(f"Database query or initial data processing failed: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

##############################################################################
# DATA PREPARATION
##############################################################################
def prepare_summary_data(df_input, logger):
    logger.info("Performing data preparations for summary analysis...")
    df = df_input.copy()

    # Define county category mapping
    category_map = {
        1: "1. No hospital",
        2: "2. Hospital, no AI/Robotics",
        3: "3. Hospital, surgical robotics",
        4: "4. Hospital, AI",
        5: "5. Hospital, AI & robotics",
        6: "6. Multiple hospitals, AI & robotics"
    }
    # Ensure county_category is numeric before mapping
    df['county_category'] = pd.to_numeric(df['county_category'], errors='coerce')
    df.dropna(subset=['county_category'], inplace=True) # Remove rows where category is not a valid number
    df['county_category'] = df['county_category'].astype(int) # Convert to int after NaN drop
    
    df['county_category_label'] = df['county_category'].map(category_map)
    
    # If any categories are not in the map (e.g. 0, 7), they will become NaN. Remove them.
    if df['county_category_label'].isnull().any():
        logger.warning(f"Found {df['county_category_label'].isnull().sum()} rows with unmapped county_category values. These rows will be removed.")
        df.dropna(subset=['county_category_label'], inplace=True)

    if df.empty:
        logger.error("DataFrame is empty after category mapping and cleaning. Cannot proceed.")
        sys.exit(1)
    logger.info(f"County categories mapped. Value counts:\n{df['county_category_label'].value_counts().sort_index()}")

    # Identify columns for numerical conversion and analysis
    # These are the actual column names in the dataframe after lowercasing
    numerical_cols = [
        'rural_raw_value', 'total_beds_in_county',
        'ratio_of_population_to_primary_care_physicians',
        'life_expectancy_raw_value', 'dv21_premature_death_ypll_rate',
        'child_mortality_raw_value', 'infant_mortality_raw_value',
        'dv15_preventable_stays_rate', 'mo1_ai_adoption_score',
        'mo2_robotics_adoption_score',
        'avg_patient_services_margin'  # <-- Add this line
    ]

    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"Column '{col}' not found in DataFrame. It will be skipped in analysis.")
            # remove from list if not found
            if col in numerical_cols: numerical_cols.remove(col)


    # For 'total_beds_in_county', counties in category "1. No hospital" likely have NaN
    # from the LEFT JOIN or 0 if we were to impute. Let's explicitly fill NaN with 0 for this variable.
    if 'total_beds_in_county' in df.columns:
        df['total_beds_in_county'].fillna(0, inplace=True)
        # Also, AI and Robotics scores should be 0 if no hospital or no tech
        if 'mo1_ai_adoption_score' in df.columns:
             df.loc[df['county_category_label'] == "1. No hospital", 'mo1_ai_adoption_score'] = df.loc[df['county_category_label'] == "1. No hospital", 'mo1_ai_adoption_score'].fillna(0)
        if 'mo2_robotics_adoption_score' in df.columns:
            df.loc[df['county_category_label'] == "1. No hospital", 'mo2_robotics_adoption_score'] = df.loc[df['county_category_label'] == "1. No hospital", 'mo2_robotics_adoption_score'].fillna(0)


    logger.info("Data preparation complete.")
    # Return the list of successfully processed numerical_cols along with the df
    return df, numerical_cols, category_map

##############################################################################
# CATEGORICAL SUMMARY ANALYSIS
##############################################################################
def perform_categorical_summary_analysis(df, numerical_cols, category_map, plot_dir, output_dir, logger):
    logger.info("="*30 + " CATEGORICAL SUMMARY ANALYSIS " + "="*30)
    
    summary_results = []
    statistical_tests_results = {}

    # Define a consistent color palette to match the map
    # These colors correspond to the legend in the map
    custom_palette = {
        "1. No hospital": "#FFFFFF",                  # White
        "2. Hospital, no AI/Robotics": "#CCCCCC",     # Light gray
        "3. Hospital, surgical robotics": "#999999",  # Medium gray
        "4. Hospital, AI": "#A1C8E8",                 # Light blue
        "5. Hospital, AI & robotics": "#4682B4",      # Medium blue 
        "6. Multiple hospitals, AI & robotics": "#003A70" # Dark blue (navy)
    }
    
    # Ensure the palette matches available categories
    ordered_categories = [category_map[i] for i in sorted(category_map.keys()) if category_map[i] in df['county_category_label'].unique()]
    category_colors = [custom_palette[cat] for cat in ordered_categories if cat in custom_palette]

    # Ensure categories are ordered correctly for tables and plots
    ordered_categories = [category_map[i] for i in sorted(category_map.keys()) if category_map[i] in df['county_category_label'].unique()]
    df['county_category_label'] = pd.Categorical(df['county_category_label'], categories=ordered_categories, ordered=True)
    df.sort_values('county_category_label', inplace=True)

    for col in numerical_cols:
        if col not in df.columns:
            logger.warning(f"Column {col} not found for summary analysis. Skipping.")
            continue
        
        logger.info(f"\n--- Analyzing Variable: {col} ---")
        
        # Descriptive Statistics
        desc_stats = df.groupby('county_category_label')[col].agg(['count', 'mean', 'std', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        desc_stats.rename(columns={'<lambda_0>': 'q1', '<lambda_1>': 'q3'}, inplace=True)
        logger.info(f"Descriptive Statistics for {col}:\n{desc_stats}")
        
        for category_label in ordered_categories:
            if category_label in desc_stats.index:
                stats = desc_stats.loc[category_label]
                summary_results.append({
                    'Variable': col,
                    'Category': category_label,
                    'N_Counties': stats['count'],
                    'Mean': stats['mean'],
                    'StdDev': stats['std'],
                    'Median': stats['median'],
                    'Q1': stats['q1'],
                    'Q3': stats['q3']
                })
            else: # Handle categories with no data for this variable after potential filtering
                 summary_results.append({
                    'Variable': col,
                    'Category': category_label,
                    'N_Counties': 0, 'Mean': np.nan, 'StdDev': np.nan,
                    'Median': np.nan, 'Q1': np.nan, 'Q3': np.nan
                })


        # Statistical Test (ANOVA or Kruskal-Wallis)
        # Prepare list of arrays for test, one array per category, excluding NaNs
        category_groups = [df[df['county_category_label'] == cat_label][col].dropna().values for cat_label in ordered_categories]
        category_groups = [group for group in category_groups if len(group) > 1] # Need at least 2 samples per group for variance/tests

        test_used = "Skipped (Insufficient data)"
        p_value = np.nan

        if len(category_groups) < 2: # Need at least two groups to compare
            logger.warning(f"  Skipping statistical test for {col}: Less than 2 valid groups.")
        else:
            # Check assumptions for ANOVA (Normality via Shapiro-Wilk, Homogeneity of variances via Levene)
            # Simplified check: if any group small or fails normality, lean towards Kruskal-Wallis
            use_anova = True
            min_samples_for_normality_check = 8 # Shapiro-Wilk needs at least 3, but more is better
            
            all_groups_normal = True
            for i, group_data in enumerate(category_groups):
                if len(group_data) >= min_samples_for_normality_check:
                    stat_shapiro, p_shapiro = shapiro(group_data)
                    if p_shapiro < 0.05:
                        all_groups_normal = False
                        logger.debug(f"  Group {ordered_categories[i]} for {col} not normal (Shapiro p={p_shapiro:.3f})")
                        break
                elif len(group_data) > 0: # Small sample, assume not normal for safety
                    all_groups_normal = False
                    logger.debug(f"  Group {ordered_categories[i]} for {col} too small for robust normality test (N={len(group_data)}), assuming non-normal.")
                    break
            
            if all_groups_normal:
                # Check homogeneity of variances if all groups seemed normal
                # Levene test is more robust to non-normality than Bartlett
                try:
                    # Filter out groups with zero variance if they exist, as Levene can fail
                    groups_for_levene = [g for g in category_groups if np.var(g) > 0]
                    if len(groups_for_levene) >=2: # Need at least two groups for Levene
                        stat_levene, p_levene = levene(*groups_for_levene)
                        if p_levene < 0.05:
                            use_anova = False
                            logger.info(f"  Variances not homogeneous for {col} (Levene p={p_levene:.3f}). Using Kruskal-Wallis.")
                    elif len(groups_for_levene) < 2 and len(category_groups) >=2 : # e.g. one group has variance, other is constant
                        use_anova = False
                        logger.info(f"  Could not robustly test homogeneity for {col} (some groups constant). Using Kruskal-Wallis.")

                except Exception as e_levene:
                    logger.warning(f"  Levene test failed for {col}: {e_levene}. Defaulting to Kruskal-Wallis.")
                    use_anova = False
            else: # Not all groups normal
                use_anova = False
                logger.info(f"  Not all groups normal for {col}. Using Kruskal-Wallis.")

            if use_anova:
                try:
                    f_stat, p_value_anova = f_oneway(*category_groups)
                    test_used = "ANOVA"
                    p_value = p_value_anova
                    logger.info(f"  ANOVA for {col}: F-statistic={f_stat:.3f}, p-value={p_value:.4f}")
                except Exception as e_anova:
                    logger.error(f"  ANOVA failed for {col}: {e_anova}. Test skipped.")
                    test_used = "ANOVA (Error)"
                    p_value = np.nan
            else: # Use Kruskal-Wallis
                # Kruskal-Wallis can handle groups with 0 variance if there are ties.
                # It requires at least two groups.
                if len(category_groups) >= 2 and all(len(g) > 0 for g in category_groups):
                    try:
                        h_stat, p_value_kruskal = kruskal(*category_groups)
                        test_used = "Kruskal-Wallis"
                        p_value = p_value_kruskal
                        logger.info(f"  Kruskal-Wallis for {col}: H-statistic={h_stat:.3f}, p-value={p_value:.4f}")
                    except Exception as e_kruskal:
                        logger.error(f"  Kruskal-Wallis failed for {col}: {e_kruskal}. Test skipped.")
                        test_used = "Kruskal-Wallis (Error)"
                        p_value = np.nan
                else:
                    logger.warning(f"  Skipping Kruskal-Wallis for {col}: Not enough groups with data.")
        
        statistical_tests_results[col] = {'test_used': test_used, 'p_value': p_value}

        # Visualization (Box Plot)
        if VISUALIZATION_AVAILABLE and df[col].notna().sum() > 0:
            plt.figure(figsize=(12, 7))
            
            # Create the boxplot with custom colors
            ax = sns.boxplot(x='county_category_label', y=col, data=df, 
                         order=ordered_categories, 
                         palette=custom_palette,
                         width=0.6)  # Slightly narrower boxes to better see dots
            
            # Add semi-transparent dots overlay
            sns.stripplot(x='county_category_label', y=col, data=df, 
                          order=ordered_categories,
                          palette=custom_palette,  # Same palette for consistency
                          size=3.5,                # Adjust dot size
                          alpha=0.3,               # Semi-transparency
                          jitter=True,             # Add slight random horizontal jitter
                          dodge=False)             # Don't dodge dots (center them)
            
            plt.title(f"Distribution of {col} by County Category\n{test_used} p-value: {p_value:.4f}" if pd.notna(p_value) 
                      else f"Distribution of {col} by County Category", fontsize=15)
            plt.ylabel(col, fontsize=12)
            plt.xlabel("County Category", fontsize=12)
            plt.xticks(rotation=25, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a light grid for readability
            plt.tight_layout()
            
            # Save the plot
            plot_path = os.path.join(plot_dir, f"boxplot_{col.replace(' ', '_')}.png")
            try:
                plt.savefig(plot_path, dpi=300)
                logger.info(f"  Box plot for {col} saved to {plot_path}")
            except Exception as e_plot:
                logger.error(f"  Failed to save box plot for {col}: {e_plot}")
            plt.close()

        # Create a log-transformed version of the boxplot if applicable
        if VISUALIZATION_AVAILABLE and df[col].notna().sum() > 0:
            min_value = df[col].min()
            
            if min_value <= 0:
                offset = abs(min_value) + 1 if min_value < 0 else 1
                log_data = df.copy()
                log_data[col] = log_data[col] + offset
                log_transform_type = "symlog"
                log_note = f" (symlog transform, offset +{offset})"
            else:
                log_data = df.copy()
                log_transform_type = "log"
                log_note = " (log transform)"
                
            plt.figure(figsize=(12, 7))
            ax = sns.boxplot(x='county_category_label', y=col, data=log_data, 
                        order=ordered_categories, 
                        palette=custom_palette,
                        width=0.6)
            sns.stripplot(x='county_category_label', y=col, data=log_data, 
                        order=ordered_categories,
                        palette=custom_palette,
                        size=3.5,
                        alpha=0.3,
                        jitter=True,
                        dodge=False)
            
            # For the patient services margin, add special features
            if col == "avg_patient_services_margin":
                # Calculate the transformed y=0 position
                zero_trans = 0 + offset if min_value <= 0 else 0
                # Draw the breakeven line
                ax.axhline(zero_trans, color='red', linestyle='--', linewidth=2, label='Breakeven (0%)')
                
                # Calculate and add mean values for each category
                for i, category in enumerate(ordered_categories):
                    # Get original (non-transformed) mean for this category
                    category_data = df[df['county_category_label'] == category][col]
                    if len(category_data) > 0:
                        mean_value = category_data.mean()
                        mean_pct_str = f"{mean_value * 100:.2f}%"
                        # Transform mean for plotting
                        trans_mean = mean_value + offset if log_transform_type == "symlog" else mean_value
                        # Place annotation at the mean value, slightly above
                        ax.text(i, trans_mean * 1.05, mean_pct_str,
                                ha='center', va='bottom',
                                color='black', fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'))
            # Add legend entry
            handles, labels = ax.get_legend_handles_labels()
            if 'Breakeven (0%)' not in labels:
                handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2))
                labels.append('Breakeven (0%)')
            ax.legend(handles, labels, loc='upper right')
            
            # Apply the appropriate scale transformation
            if log_transform_type == "log":
                ax.set_yscale('log')
            else:
                ax.set_yscale('symlog')
            
            plt.title(f"Distribution of {col} by County Category{log_note}\n{test_used} p-value: {p_value:.4f}" 
                    if pd.notna(p_value) else f"Distribution of {col} by County Category{log_note}", 
                    fontsize=15)
            
            # For patient services margin, adjust the y-label to emphasize percentages
            if col == "avg_patient_services_margin":
                plt.ylabel(f"Patient Services Margin (%) - Log Scale", fontsize=12)
            else:
                plt.ylabel(f"{col} (Log Scale)", fontsize=12)
                
            plt.xlabel("County Category", fontsize=12)
            plt.xticks(rotation=25, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            log_plot_path = os.path.join(plot_dir, f"boxplot_log_{col.replace(' ', '_')}.png")
            try:
                plt.savefig(log_plot_path, dpi=300)
                logger.info(f"  Log-transformed box plot for {col} saved to {log_plot_path}")
            except Exception as e_plot:
                logger.error(f"  Failed to save log-transformed box plot for {col}: {e_plot}")
            plt.close()

    # Format summary results into a table
    summary_df = pd.DataFrame(summary_results)
    summary_pivot_table = summary_df.pivot_table(
        index='Category', 
        columns='Variable', 
        values=['N_Counties', 'Mean', 'StdDev', 'Median', 'Q1', 'Q3']
    )
    
    # Reorder columns for better readability: N, Mean, Std for Var1, then N, Mean, Std for Var2 etc.
    # Or simpler: group by variable first.
    # Let's create a table where columns are Variable and rows are Category, with multi-level stats as sub-columns
    # Or simpler: Variable as primary column, stats as sub-columns
    
    # Create a more readable format: Category as rows, Variables as columns, cell contains "Mean (SD)" or "Median (IQR)"
    final_summary_table = pd.DataFrame(index=ordered_categories)
    for col in numerical_cols:
        if col not in df.columns: continue
        
        means = summary_df[summary_df['Variable'] == col].set_index('Category')['Mean']
        stds = summary_df[summary_df['Variable'] == col].set_index('Category')['StdDev']
        final_summary_table[f'{col} (Mean ± SD)'] = means.map('{:.2f}'.format) + " ± " + stds.map('{:.2f}'.format)
        
        medians = summary_df[summary_df['Variable'] == col].set_index('Category')['Median']
        q1s = summary_df[summary_df['Variable'] == col].set_index('Category')['Q1']
        q3s = summary_df[summary_df['Variable'] == col].set_index('Category')['Q3']
        final_summary_table[f'{col} (Median (IQR))'] = medians.map('{:.2f}'.format) + " (" + q1s.map('{:.2f}'.format) + " - " + q3s.map('{:.2f}'.format) + ")"
        
        # Add N_Counties
        n_counts = summary_df[summary_df['Variable'] == col].set_index('Category')['N_Counties']
        final_summary_table[f'{col} (N Counties)'] = n_counts.astype(int)


    # Add statistical test results as a final row
    stat_test_row_data = {}
    for col_name in numerical_cols:
        if col_name in statistical_tests_results:
            res = statistical_tests_results[col_name]
            stat_test_row_data[f'{col_name} (Mean ± SD)'] = f"{res['test_used']}: p={res['p_value']:.4f}" if pd.notna(res['p_value']) else res['test_used']
            stat_test_row_data[f'{col_name} (Median (IQR))'] = "" # Keep this column for alignment
            stat_test_row_data[f'{col_name} (N Counties)'] = ""    # Keep this column for alignment
        else:
            stat_test_row_data[f'{col_name} (Mean ± SD)'] = "Not tested"
            stat_test_row_data[f'{col_name} (Median (IQR))'] = ""
            stat_test_row_data[f'{col_name} (N Counties)'] = ""

    # Ensure all columns exist before trying to create the Series
    for col_header in final_summary_table.columns:
        if col_header not in stat_test_row_data:
            stat_test_row_data[col_header] = "" # default if a variable was skipped

    stat_test_series = pd.Series(stat_test_row_data, name="Statistical Test (Across Categories)")
    final_summary_table = pd.concat([final_summary_table, pd.DataFrame(stat_test_series).T])


    logger.info(f"\nFinal Summary Table:\n{final_summary_table.to_string()}")
    summary_csv_path = os.path.join(output_dir, "county_category_summary_table.csv")
    try:
        final_summary_table.to_csv(summary_csv_path)
        logger.info(f"Summary table saved to {summary_csv_path}")
    except Exception as e_csv:
        logger.error(f"Failed to save summary_df CSV: {e_csv}")

    return final_summary_table


##############################################################################
# MAIN SCRIPT EXECUTION
##############################################################################
def main():
    logger = setup_logger()
    logger.info("Starting County Category Summary Analysis Script.")

    output_dir = "county_category_summary_output"
    plot_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    
    warnings.filterwarnings("ignore", category=UserWarning) # General user warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) # Common in stats calcs with NaNs
    warnings.filterwarnings("ignore", category=FutureWarning) 

    try:
        engine = connect_to_database(logger)
        df_raw = fetch_data_for_summary_analysis(engine, logger)
        
        if df_raw.empty:
            logger.error("No data fetched. Exiting.")
            sys.exit(1)

        df_prepared, numerical_cols_for_analysis, category_map = prepare_summary_data(df_raw, logger)

        if df_prepared.empty:
            logger.error("Data became empty after preparation. Exiting.")
            sys.exit(1)
        
        if not numerical_cols_for_analysis:
            logger.error("No numerical columns identified for analysis after preparation. Exiting.")
            sys.exit(1)

        perform_categorical_summary_analysis(df_prepared, numerical_cols_for_analysis, category_map, plot_dir, output_dir, logger)

    except Exception as e_main:
        logger.error(f"An error occurred in the main execution block: {e_main}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'engine' in locals() and engine:
            engine.dispose()
            logger.info("Database engine disposed.")

    logger.info("Script finished. Check logs and output directory for details.")

if __name__ == "__main__":
    main()