#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
"""
Hospital System Technology Adoption Analysis

This script analyzes the relationship between hospital system characteristics and technology adoption.
It focuses on several key research questions:
1. Do system-affiliated hospitals have higher technology adoption rates than independent hospitals?
2. Does health system size correlate with technology adoption levels?
3. Is there a relationship between system size and capital expenditure per square foot?
4. Does patient volume influence technology adoption?

The script connects to a PostgreSQL database, extracts hospital survey data,
performs statistical analysis, and generates visualizations of the results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sqlalchemy import create_engine
import logging

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================

# --- Output Configuration ---
# Create directories to store results and plots
os.makedirs('results', exist_ok=True)  # Creates directory if it doesn't exist, otherwise does nothing
os.makedirs('plots', exist_ok=True)    # Will store all generated visualizations

# --- Logging Configuration ---
# Set up logging to file and console
log_file = os.path.join('results', 'analysis.log')
logging.basicConfig(
    level=logging.INFO,  # Sets the threshold for logging messages
    format='%(asctime)s - %(levelname)s - %(message)s',  # Timestamp, log level, and message
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Write logs to file (overwrite existing)
        logging.StreamHandler()  # Also display logs in console/terminal
    ]
)
logging.info("Starting analysis script...")


# =============================================================================
# 2. DATABASE CONNECTION
# =============================================================================

# --- Database Credentials ---
# Best practice: use environment variables for sensitive data
host = 'localhost'  # Database server address
database = 'Research_TEST'  # Name of the database containing hospital survey data
user = 'postgres'           # Database username for authentication
password = os.getenv("POSTGRESQL_KEY")  # Password retrieved from environment variables for security

# Check if the password environment variable is set
if not password:
    logging.error("POSTGRESQL_KEY environment variable not set. Exiting.")
    exit()

try:
    # Create a SQLAlchemy engine to connect to PostgreSQL
    connection_string = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"
    engine = create_engine(connection_string)
    logging.info(f"Successfully created connection engine for database '{database}' on host '{host}'.")
except Exception as e:
    logging.error(f"Failed to create database engine: {e}")
    exit()


# =============================================================================
# 3. FETCH AND PREPARE DATA
# =============================================================================

def fetch_and_prepare_data(engine):
    """
    Fetches raw hospital data from Postgres, cleans it, and engineers
    new features required for the analysis.
    
    Parameters:
    - engine: SQLAlchemy engine for database connection
    
    Returns:
    - Pandas DataFrame with cleaned and transformed hospital data
    """
    logging.info("Fetching data from 'aha_survey_data' table...")
    
    # Define all columns of interest from the hospital survey data
    fields_to_fetch = [
        'id', 'sysname',       # Hospital identifier and system name
        'adjpd',               # Adjusted patient days (measure of hospital volume)
        'bdtot',               # Total beds
        'ceamt',               # Capital expenditure amount
        'gfeet',               # Gross square feet (hospital size)
        # Robotics fields (binary adoption indicators)
        'robohos',             # Hospital-wide robotics system
        'robosys',             # Surgical robotics system
        'roboven',             # Vendor robotics system
        # AI fields (binary adoption indicators)
        'wfaiart',             # AI for automated report transcription
        'wfaiss',              # AI for smart scheduling
        'wfaipsn',             # AI for personalized patient notifications
        'wfaippd',             # AI for population disease prediction
        'wfaioacw'             # AI for optimization of administrative/clinical workflows
    ]
    
    query = f"SELECT {', '.join(fields_to_fetch)} FROM public.aha_survey_data"
    
    try:
        df = pd.read_sql(query, engine)
        logging.info(f"Successfully fetched {len(df)} records.")
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

    # --- Data Cleaning and Type Conversion ---
    # Convert all numeric fields to proper data types, with error handling
    numeric_cols = fields_to_fetch[2:] # All columns except id and sysname
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 'coerce' converts invalid values to NaN

    logging.info("Initial data types and missing values:\n" + str(df.info()))

    # --- Feature Engineering ---
    logging.info("Performing feature engineering...")
    
    # Q1: Is the hospital part of a system?
    # Create binary indicator (1=yes, 0=no) based on whether system name is available
    df['is_in_system'] = df['sysname'].notna().astype(int)

    # Q2: Calculate the size of each health system
    # Count how many hospitals are in each system
    # We use transform to broadcast the count to every row of the group
    df['system_size'] = df.groupby('sysname')['id'].transform('count')
    # Fill NaN for independent hospitals so the column is complete
    df['system_size'].fillna(0, inplace=True) 

    # Q3: Calculate CAPEX per square foot, handling division by zero
    # This metric shows investment intensity relative to facility size
    df['capex_per_sqft'] = np.where(df['gfeet'] > 0, df['ceamt'] / df['gfeet'], np.nan)
    
    # Create aggregate technology scores for broader analysis
    # Sum the binary indicators to create composite adoption scores
    ai_cols = ['wfaiart', 'wfaiss', 'wfaipsn', 'wfaippd', 'wfaioacw']
    robotics_cols = ['robohos', 'robosys', 'roboven']
    
    # These scores represent the breadth of technology adoption (higher = more technologies)
    df['ai_adoption_score'] = df[ai_cols].sum(axis=1)
    df['robotics_adoption_score'] = df[robotics_cols].sum(axis=1)

    logging.info("Feature engineering complete. New columns added: 'is_in_system', 'system_size', 'capex_per_sqft', 'ai_adoption_score', 'robotics_adoption_score'.")
    logging.info(f"Dataset contains {df['is_in_system'].sum()} hospitals in a system and {len(df) - df['is_in_system'].sum()} independent hospitals.")
    
    return df


# =============================================================================
# 4. ANALYSIS & VISUALIZATION HELPERS
# =============================================================================

def perform_linear_regression(df, predictor_col, outcome_col):
    """
    Runs a simple OLS regression for one predictor and one outcome.
    
    Parameters:
    - df: DataFrame containing the data
    - predictor_col: Name of the independent/predictor variable column
    - outcome_col: Name of the dependent/outcome variable column
    
    Returns:
    - Dictionary with key statistics (R-squared, p-value, coefficient, sample size)
    """
    # Drop rows with missing values for the specific variables in this regression
    subset_df = df[[predictor_col, outcome_col]].dropna()
    
    if len(subset_df) < 10: # Ensure there's enough data to run a model
        return {'r_squared': np.nan, 'p_value': np.nan, 'beta': np.nan, 'n_obs': len(subset_df)}

    y = subset_df[outcome_col]     # Dependent variable (technology adoption)
    X = subset_df[predictor_col]   # Independent variable (system characteristic)
    X = sm.add_constant(X)         # Add an intercept term to the model
    
    # Fit the Ordinary Least Squares regression model
    model = sm.OLS(y, X).fit()
    
    # Return key statistics from the regression
    return {
        'r_squared': model.rsquared,              # Proportion of variance explained
        'p_value': model.pvalues[predictor_col],  # Statistical significance
        'beta': model.params[predictor_col],      # Effect size (coefficient)
        'n_obs': model.nobs                       # Sample size
    }

def create_scatterplot(df, x_col, y_col, file_path):
    """
    Creates and saves a publication-ready scatterplot with a regression line.
    
    Parameters:
    - df: DataFrame containing the data
    - x_col: Name of the column for the x-axis
    - y_col: Name of the column for the y-axis
    - file_path: Where to save the generated plot
    """
    # Drop NaNs for plotting to avoid warnings and errors
    plot_df = df[[x_col, y_col]].dropna()
    
    if len(plot_df) < 10:
        logging.warning(f"Skipping plot for {y_col} vs {x_col} due to insufficient data ({len(plot_df)} points).")
        return

    # Create the figure with appropriate size for publication
    plt.figure(figsize=(10, 6))
    
    # Create a scatterplot with regression line using seaborn
    sns.regplot(data=plot_df, x=x_col, y=y_col, 
                scatter_kws={'alpha':0.3},         # Transparency for better visibility with overlapping points
                line_kws={'color':'red', 'linewidth':2})  # Highlight the trend line
    
    # Add labels and formatting
    plt.title(f'Relationship between {y_col} and {x_col}', fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # Save the figure at publication quality (300 DPI)
    plt.savefig(file_path, dpi=300)
    plt.close()  # Close to free memory and avoid display in notebooks
    logging.info(f"Saved scatterplot to {file_path}")


# =============================================================================
# 5. RUN ANALYSIS & GENERATE OUTPUTS
# =============================================================================

def run_analysis(df):
    """
    Main function to run all analyses based on the research questions.
    
    Parameters:
    - df: DataFrame containing the prepared hospital data
    """
    if df is None:
        logging.error("DataFrame is None. Cannot run analysis.")
        return

    # Define the technology adoption columns to be used as outcomes
    # These are the dependent variables we want to explain
    tech_cols = [
        # Individual robotics technologies
        'robohos', 'robosys', 'roboven',
        # Individual AI technologies
        'wfaiart', 'wfaiss', 'wfaipsn', 'wfaippd', 'wfaioacw',
        # Composite adoption scores
        'robotics_adoption_score', 'ai_adoption_score'
    ]
    
    # Container for all regression results
    all_results = []

    # --- Question 1: System vs. Independent Hospitals ---
    # Are hospitals in systems more likely to adopt technologies?
    logging.info("\n" + "="*50 + "\nANALYSIS 1: System vs. Independent Hospital Tech Adoption\n" + "="*50)
    
    # Summary Statistics - Compare means between system and non-system hospitals
    summary_q1 = df.groupby('is_in_system')[tech_cols].mean().round(3)
    summary_q1.index = ['Independent', 'In-System']  # Rename indices for clarity
    logging.info("Mean technology adoption scores (0=No, 1=Yes):\n" + str(summary_q1.T))
    
    # Regression Analysis - Quantify the system effect for each technology
    for tech_col in tech_cols:
        results = perform_linear_regression(df, 'is_in_system', tech_col)
        results.update({'analysis': 'System vs. Independent', 'predictor': 'is_in_system', 'outcome': tech_col})
        all_results.append(results)

    # --- Question 2: Health System Size and Tech Adoption ---
    # Does being in a larger system correlate with higher tech adoption?
    logging.info("\n" + "="*50 + "\nANALYSIS 2: Health System Size and Tech Adoption\n" + "="*50)
    
    # Filter for only hospitals that are part of a system
    system_hospitals = df[df['is_in_system'] == 1].copy()
    
    for tech_col in tech_cols:
        # Run regression for each technology
        results = perform_linear_regression(system_hospitals, 'system_size', tech_col)
        results.update({'analysis': 'System Size vs. Tech', 'predictor': 'system_size', 'outcome': tech_col})
        all_results.append(results)
        
        # Create visualization of the relationship
        plot_path = os.path.join('plots', f'scatterplot_system_size_vs_{tech_col}.png')
        create_scatterplot(system_hospitals, 'system_size', tech_col, plot_path)

    # --- Question 3: System Size and CAPEX per Square Foot ---
    # Do larger systems invest more in capital expenditures per facility size?
    logging.info("\n" + "="*50 + "\nANALYSIS 3: Health System Size and CAPEX per SqFt\n" + "="*50)
    
    # Run regression for capital expenditure
    results = perform_linear_regression(system_hospitals, 'system_size', 'capex_per_sqft')
    results.update({'analysis': 'System Size vs. CAPEX', 'predictor': 'system_size', 'outcome': 'capex_per_sqft'})
    all_results.append(results)
    
    # Create visualization
    plot_path = os.path.join('plots', 'scatterplot_system_size_vs_capex_per_sqft.png')
    create_scatterplot(system_hospitals, 'system_size', 'capex_per_sqft', plot_path)

    # --- Question 4: Patient Volume and Tech Adoption ---
    # Does higher patient volume correlate with more technology adoption?
    logging.info("\n" + "="*50 + "\nANALYSIS 4: Patient Volume (adjpd) and Tech Adoption\n" + "="*50)
    
    for tech_col in tech_cols:
        # Run regression for each technology
        results = perform_linear_regression(df, 'adjpd', tech_col)
        results.update({'analysis': 'Patient Volume vs. Tech', 'predictor': 'adjpd', 'outcome': tech_col})
        all_results.append(results)
        
        # Create visualization
        plot_path = os.path.join('plots', f'scatterplot_adjpd_vs_{tech_col}.png')
        create_scatterplot(df, 'adjpd', tech_col, plot_path)

    # --- Compile and Save Results ---
    # Convert all regression results to a DataFrame for export
    results_df = pd.DataFrame(all_results)
    results_df = results_df[['analysis', 'predictor', 'outcome', 'beta', 'r_squared', 'p_value', 'n_obs']]
    
    # Add significance stars for clarity in reporting
    results_df['significance'] = results_df['p_value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    )
    
    # Save to CSV for further analysis or reporting
    csv_path = os.path.join('results', 'hospital_tech_adoption_analysis_results.csv')
    results_df.to_csv(csv_path, index=False)
    logging.info(f"\nAll regression results saved to {csv_path}")
    
    # Print formatted results to log/console
    logging.info("\n" + "="*70 + "\nSUMMARY OF REGRESSION RESULTS\n" + "="*70)
    logging.info("Significance codes: *** p<0.001, ** p<0.01, * p<0.05\n")
    with pd.option_context('display.max_rows', None, 'display.width', 1000):
        logging.info(results_df)


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Step 1: Fetch and prepare data from the database
    hospital_df = fetch_and_prepare_data(engine)
    
    # Step 2: Run the full analysis pipeline
    if hospital_df is not None:
        run_analysis(hospital_df)
        logging.info("\nAnalysis script finished successfully.")
    else:
        logging.error("Analysis could not be performed due to data loading issues.")