#!/usr/bin/env python3
# ==============================================================================
#  geospatial_hospital_ai_robotics_analysis.py
#  Author:  Aaron Johnson
#  Last updated: 2025-07-14
#  VERSION 14.0 - Corrects the k-NN graph visualization logic to prevent
#                 spurious connections to non-contiguous states by filtering
#                 the dataset before graph creation.
#
#  VERSION 12.0 - This version incorporates a new k-Nearest Neighbors (k-NN)
#                 graph analysis to visualize the network connectivity
#                 between hospitals based on geographic proximity.
#
#  VERSION 11.0 - This version incorporates feedback to refine visualizations
#                 and strengthen the core regression model.
#                 1. Adds a 3rd Lorenz curve for baseline hospital access.
#                 2. Annotates regression plot with coefficients and p-values.
#                 3. Fixes the centering of the contiguous US LISA map.
#                 4. Implements a new regression model using composite health
#                    indices and controls for population and census division.
#
#  End-to-end workflow analyzing spatial patterns of Generative AI and
#  robotics adoption in US hospitals (AHA 2024). This script performs:
#   1. Proximity analysis with robust outlier removal.
#   2. Population coverage analysis for AI/robotics hospitals.
#   3. Inequality analysis using Lorenz curves and Gini coefficients.
#   4. Regression analysis linking technology access to Years of Potential Life Lost (YPLL).
#   5. Hot spot (LISA) analysis to find spatial clusters of adoption.
#   6. k-NN graph analysis to visualize hospital network structure.
#   7. Generates publication-quality maps and figures.
# ==============================================================================
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import traceback

# --- Core Libraries ---
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine, text
from tqdm import tqdm

# --- Analysis Libraries ---
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample
from numpy import trapz
import statsmodels.api as sm
import statsmodels.formula.api as smf # For detailed regression output
import statsmodels.stats.weightstats as wstats # Add this import for weighted stats

# --- Visualization Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Optional Geospatial Libraries (with graceful failure) ---
try:
    import contextily as cx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    print("Warning: 'contextily' not found. Basemaps will not be added to maps.")

try:
    import esda
    import libpysal
    from splot.esda import lisa_cluster
    PYSAL_AVAILABLE = True
except ImportError:
    PYSAL_AVAILABLE = False
    print("Warning: 'pysal', 'esda', 'splot' not found. Hot spot analysis will be skipped.")

# Use Albers Equal Area projection specifically designed for US maps
from matplotlib.figure import Figure
try:
    from cartopy.crs import AlbersEqualArea
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: 'cartopy' not found. Some map projections will not be available.")

# ======================= CONFIGURATION ==========================
BASE_DIR = Path(__file__).resolve().parent
SHAPE_DIR = BASE_DIR / "shapefiles"
FIG_DIR = BASE_DIR / "figures"
LOG_FILE = BASE_DIR / "analysis_log.txt"
CX_CACHE = BASE_DIR / "tile_cache"

COUNTY_FILE = SHAPE_DIR / "tl_2024_us_county.shp"
POP_FILE = SHAPE_DIR / "USA_BlockGroups_2020Pop.geojson"

DRIVE_MINUTES = 30
AVG_SPEED_MPH = 40.0
RADIUS_MILES = AVG_SPEED_MPH * DRIVE_MINUTES / 60.0
EARTH_RADIUS_MILES = 3958.8

WGS84_CRS = "EPSG:4326"
AEA_CRS = "EPSG:5070"

# ======================= SETUP ================================
def setup_environment():
    """Create necessary directories and configure logging."""
    for p in (SHAPE_DIR, FIG_DIR, CX_CACHE):
        p.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("=" * 60)
    logging.info(f"RUN STARTED: {datetime.now().isoformat()}")
    logging.info("=" * 60)

# ======================= DATA LOADING & PREPROCESSING =======================
def connect_to_db():
    """Establishes a connection to the PostgreSQL database."""
    try:
        db_password = os.getenv("POSTGRESQL_KEY")
        if not db_password:
            raise ValueError("POSTGRESQL_KEY environment variable not set.")
        engine = create_engine(
            f"postgresql+psycopg2://{os.getenv('PGUSER', 'postgres')}:{db_password}@{os.getenv('PGHOST', 'localhost')}/{os.getenv('PGDATABASE', 'Research_TEST')}",
            pool_pre_ping=True, connect_args={"connect_timeout": 10},
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info(f"Postgres connection successful: {engine.url.host}/{engine.url.database}")
        return engine
    except Exception as e:
        logging.exception("FATAL: Failed to connect to PostgreSQL database.")
        sys.exit(1)

def load_hospital_data(engine):
    """Loads hospital data from the PostgreSQL view."""
    query = """
    SELECT
        hospital_id, mname, address, city, state, zipcode, county_fips,
        latitude, longitude, bsc, sysname, robohos, robosys, roboven, adjpd,
        ftemd, ftern, wfaipsn, wfaippd, wfaiss, wfaiart, wfaioacw, gfeet, ceamt
    FROM vw_geocoded_hospitals;
    """
    try:
        df = pd.read_sql(query, engine)
        logging.info(f"Successfully loaded {len(df)} records from vw_geocoded_hospitals.")
        return df
    except Exception as e:
        logging.exception("FATAL: Error executing query on hospital data.")
        sys.exit(1)

def load_population_data(pop_filepath):
    """Loads and preprocesses population data from a GeoJSON file."""
    logging.info(f"Attempting to load population data from: {pop_filepath.name}")
    if not pop_filepath.exists():
        logging.error(f"FATAL: Population GeoJSON file not found at: {pop_filepath}")
        return None
    try:
        pop_gdf = gpd.read_file(pop_filepath)
        logging.info(f"Successfully loaded {len(pop_gdf)} records from {pop_filepath.name}")
        if 'population' in pop_gdf.columns and 'POPULATION' not in pop_gdf.columns:
            logging.info("Renaming 'population' column to 'POPULATION' for consistency.")
            pop_gdf.rename(columns={'population': 'POPULATION'}, inplace=True)

        if 'POPULATION' not in pop_gdf.columns:
            logging.error("FATAL: 'POPULATION' column not found in GeoJSON.")
            return None

        pop_gdf['POPULATION'] = pd.to_numeric(pop_gdf['POPULATION'], errors='coerce').fillna(0)
        pop_gdf = pop_gdf[pop_gdf['POPULATION'] > 0] # Remove zero-pop block groups
        logging.info(f"{len(pop_gdf)} block groups remain after filtering for non-zero population.")
        return pop_gdf
    except Exception as e:
        logging.error(f"Failed to load or process population GeoJSON file: {e}", exc_info=True)
        return None

def preprocess_data(df):
    """Cleans data, creates analysis columns, filters to US states, and returns a GeoDataFrame."""
    logging.info("Preprocessing data...")
    df.dropna(subset=["latitude", "longitude"], inplace=True)
    df = df[(pd.to_numeric(df['latitude'], errors='coerce').notna()) & (pd.to_numeric(df['longitude'], errors='coerce').notna())]
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    logging.info(f"{len(df)} hospitals remaining after cleaning coordinates.")

    ai_cols = ["wfaipsn", "wfaippd", "wfaiss", "wfaiart", "wfaioacw"]
    for col in ai_cols + ['robohos', 'bsc']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['ai_flag'] = (df[ai_cols].max(axis=1) > 0).astype(int)
    df['ai_intensity'] = df[ai_cols].sum(axis=1)
    df['robo_flag'] = (df['robohos'] > 0).astype(int)

    def get_tech_type(row):
        if row['ai_flag'] == 1 and row['robo_flag'] == 1: return 'AI & Robotics'
        elif row['ai_flag'] == 1: return 'AI Only'
        elif row['robo_flag'] == 1: return 'Robotics Only'
        else: return 'Neither'
    df['tech_type'] = df.apply(get_tech_type, axis=1)

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=WGS84_CRS
    )
    logging.info("Created initial GeoDataFrame with all hospital locations.")

    logging.info("\n--- Filtering out distant US territories ---")
    original_count = len(gdf)
    excluded_territories = ['PR', 'GU', 'AS', 'VI', 'MP']
    gdf_filtered = gdf[~gdf['state'].isin(excluded_territories)].copy()

    num_removed = original_count - len(gdf_filtered)
    if num_removed > 0:
        removed_hospitals = gdf[gdf['state'].isin(excluded_territories)]
        logging.info(f"Removed {num_removed} hospitals located in distant US territories.")
        logging.info(f"Removed facilities are in territories: {list(removed_hospitals['state'].unique())}")
    else:
        logging.info("No hospitals found in excluded US territories.")

    logging.info(f"Dataset for analysis (including AK & HI) now contains {len(gdf_filtered)} hospitals.")

    return gdf_filtered

# ======================= ANALYSIS FUNCTIONS ==========================
def calculate_proximity_metrics(gdf):
    """
    Calculates proximity metrics and removes statistical outliers based on distance.
    This method preserves valid data from Alaska and Hawaii.
    """
    logging.info("\n" + "="*20 + " PROXIMITY ANALYSIS & OUTLIER REMOVAL " + "="*20)
    if len(gdf) < 4:
        logging.warning("Not enough hospitals (< 4) to calculate k=3 neighbors. Skipping.")
        gdf['nearest_miles'], gdf['k3_avg_miles'] = np.nan, np.nan
        return gdf

    coords_rad = np.deg2rad(gdf[['latitude', 'longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')
    distances, _ = tree.query(coords_rad, k=4)
    distances_miles = distances * EARTH_RADIUS_MILES
    gdf['nearest_miles'] = distances_miles[:, 1]
    gdf['k3_avg_miles'] = distances_miles[:, 1:4].mean(axis=1)

    logging.info("Calculated initial proximity metrics for all hospitals.")
    logging.info(f"Pre-filtering stats: Mean distance={gdf['nearest_miles'].mean():.2f}, Max distance={gdf['nearest_miles'].max():.2f}")

    Q1 = gdf['nearest_miles'].quantile(0.25)
    Q3 = gdf['nearest_miles'].quantile(0.75)
    IQR = Q3 - Q1
    iqr_multiplier = 3.0 # A generous multiplier to keep legitimate remote hospitals
    upper_bound = Q3 + (IQR * iqr_multiplier)

    logging.info(f"IQR-based outlier detection: Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
    logging.info(f"Calculated upper bound for nearest distance: {upper_bound:.2f} miles.")

    outliers = gdf[gdf['nearest_miles'] > upper_bound]

    if not outliers.empty:
        logging.warning(
            f"IDENTIFIED AND REMOVED {len(outliers)} distance outliers using IQR method (nearest_miles > {upper_bound:.2f} miles)."
        )
        logging.warning("Removed outlier details:\n"
                     f"{outliers[['hospital_id', 'mname', 'city', 'state', 'nearest_miles']].to_string(index=False)}")
        gdf_filtered = gdf[gdf['nearest_miles'] <= upper_bound].copy()
        logging.info(f"{len(gdf_filtered)} hospitals remain after outlier removal.")
    else:
        logging.info("No significant distance outliers found using the IQR method.")
        gdf_filtered = gdf.copy()

    final_gdf = gdf_filtered
    logging.info("\n--- Proximity Metrics on Final, Cleaned Dataset ---")
    logging.info(f"Mean distance to nearest hospital: {final_gdf['nearest_miles'].mean():.2f} miles (SD: {final_gdf['nearest_miles'].std():.2f})")
    logging.info(f"Median distance to nearest hospital: {final_gdf['nearest_miles'].median():.2f} miles")
    logging.info(f"Max distance in final dataset: {final_gdf['nearest_miles'].max():.2f} miles")
    logging.info(f"Mean avg distance to 3 nearest: {final_gdf['k3_avg_miles'].mean():.2f} miles (SD: {final_gdf['k3_avg_miles'].std():.2f})")

    return final_gdf

def analyze_population_coverage(hospital_gdf, pop_gdf):
    """Analyzes population within a 30-min drive of AI/Robotics hospitals."""
    logging.info("\n" + "="*20 + " POPULATION COVERAGE ANALYSIS " + "="*20)
    if pop_gdf is None:
        logging.error("Population data not available. Skipping population coverage analysis.")
        return

    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}, skipping coverage map.")
        return

    try:
        hosp_proj = hospital_gdf.to_crs(AEA_CRS)
        pop_proj = pop_gdf.to_crs(AEA_CRS)
        total_population = pop_proj['POPULATION'].sum()

        if total_population == 0:
            logging.error("Total population is zero. Check input data. Aborting population analysis.")
            return

        logging.info(f"Total US population from block groups: {total_population:,.0f}")
        logging.info(f"Using {DRIVE_MINUTES}-minute drive time proxy ({RADIUS_MILES:.1f} mile buffer).")

        def calculate_coverage(tech_gdf, tech_name):
            if tech_gdf.empty:
                logging.warning(f"No hospitals found for '{tech_name}'. Coverage is 0.")
                return None, 0, 0.0
            buffer_radius_meters = RADIUS_MILES * 1609.34
            
            coverage_area = gpd.GeoSeries(tech_gdf.buffer(buffer_radius_meters), crs=tech_gdf.crs).union_all()
            
            pop_proj['centroid'] = pop_proj.geometry.centroid
            covered_blocks = pop_proj.set_geometry('centroid')[pop_proj.set_geometry('centroid').within(coverage_area)]
            covered_pop = covered_blocks['POPULATION'].sum()
            percentage_covered = (covered_pop / total_population) * 100 if total_population > 0 else 0
            logging.info(f"[{tech_name}] Coverage: {covered_pop:,.0f} people ({percentage_covered:.2f}% of total).")
            return coverage_area, covered_pop, percentage_covered

        ai_area, _, _ = calculate_coverage(hosp_proj[hosp_proj['ai_flag'] == 1], 'AI-Enabled')
        robo_area, _, _ = calculate_coverage(hosp_proj[hosp_proj['robo_flag'] == 1], 'Robotics')

        us_counties = gpd.read_file(COUNTY_FILE).to_crs(AEA_CRS)
        us_contig = us_counties[~us_counties['STATEFP'].isin(['02', '15', '60', '66', '69', '72', '78'])]
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        us_contig.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)
        if ai_area: gpd.GeoSeries([ai_area], crs=AEA_CRS).plot(ax=ax, color='blue', alpha=0.5, label=f'AI Hospital {DRIVE_MINUTES}min Coverage')
        if robo_area: gpd.GeoSeries([robo_area], crs=AEA_CRS).plot(ax=ax, color='red', alpha=0.5, label=f'Robotics Hospital {DRIVE_MINUTES}min Coverage')
        ax.set_title(f'Population Coverage by AI & Robotics Hospitals ({DRIVE_MINUTES}-Min Drive Time Proxy)', fontsize=16)
        ax.set_axis_off()
        ax.legend(loc='lower left')

        if CONTEXTILY_AVAILABLE:
            cx.add_basemap(ax, crs=AEA_CRS, source=cx.providers.CartoDB.Positron, zoom='auto')

        fig_path = FIG_DIR / "population_coverage_map.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved population coverage map to {fig_path}")
        plt.close(fig)

    except Exception as e:
        logging.error(f"Error during population coverage analysis: {e}", exc_info=True)

def analyze_inequality_with_lorenz(hospital_gdf, pop_gdf):
    """
    Computes and plots Lorenz curves and Gini coefficients to measure
    inequality of geographic access to AI, Robotics, and All hospitals.
    """
    logging.info("\n" + "="*20 + " INEQUALITY (LORENZ CURVE) ANALYSIS " + "="*20)
    if pop_gdf is None:
        logging.error("Population data not available. Skipping Lorenz curve analysis.")
        return

    # --- 1. Prepare data ---
    hospitals_ai = hospital_gdf[hospital_gdf['ai_flag'] == 1].copy()
    hospitals_robo = hospital_gdf[hospital_gdf['robo_flag'] == 1].copy()
    hospitals_all = hospital_gdf.copy()

    if hospitals_ai.empty or hospitals_robo.empty:
        logging.warning("No AI or Robotics hospitals found. Cannot perform inequality analysis.")
        return

    # --- 2. Compute shortest distance from each CBG to nearest tech hospital ---
    logging.info("Projecting CBGs to AEA for accurate centroid calculation.")
    cbg_gdf_proj = pop_gdf.to_crs(AEA_CRS)
    cbg_centroids_proj = cbg_gdf_proj.geometry.centroid
    
    cbg_centroids_wgs84 = cbg_centroids_proj.to_crs(WGS84_CRS)
    cbg_coords_rad = np.deg2rad(cbg_centroids_wgs84.apply(lambda p: (p.y, p.x)).tolist())
    
    def get_nearest_distances(tech_gdf, cbg_coords_rad, tech_name):
        """Calculates shortest distance from CBGs to a set of hospitals."""
        logging.info(f"Calculating shortest distance to {len(tech_gdf)} '{tech_name}' hospitals for {len(cbg_coords_rad)} CBGs.")
        tech_coords_rad = np.deg2rad(tech_gdf[['latitude', 'longitude']].values)
        tree = BallTree(tech_coords_rad, metric='haversine')
        distances_rad, _ = tree.query(cbg_coords_rad, k=1)
        distances_miles = distances_rad.flatten() * EARTH_RADIUS_MILES
        
        distances_miles[distances_miles == 0] = 0.1
        logging.info(f"--> Applied 0.1 mile floor to {np.sum(distances_miles == 0.1)} co-located block groups.")
        
        return distances_miles

    cbg_gdf = pop_gdf.copy()
    cbg_gdf['dist_ai'] = get_nearest_distances(hospitals_ai, cbg_coords_rad, 'AI')
    cbg_gdf['dist_robo'] = get_nearest_distances(hospitals_robo, cbg_coords_rad, 'Robotics')
    cbg_gdf['dist_all'] = get_nearest_distances(hospitals_all, cbg_coords_rad, 'All')

    # --- 2. Decile Analysis (Reviewer Request A) ---
    logging.info("\n" + "-"*15 + " Decile Analysis " + "-"*15)
    logging.info("Analyzing travel burden by population decile.")

    def create_decile_table(df, dist_col, pop_col):
        """Creates a population-weighted decile analysis table."""
        df_sorted = df.sort_values(dist_col)
        df_sorted['cum_pop_share'] = df_sorted[pop_col].cumsum() / df_sorted[pop_col].sum()
        df_sorted['decile'] = pd.qcut(df_sorted['cum_pop_share'], 10, labels=range(1, 11))
        df_sorted['travel_burden'] = df_sorted[dist_col] * df_sorted[pop_col]
        total_burden = df_sorted['travel_burden'].sum()

        def weighted_avg(group, avg_name, weight_name):
            d = group[avg_name]
            w = group[weight_name]
            return (d * w).sum() / w.sum()

        decile_summary = df_sorted.groupby('decile').apply(
            lambda g: pd.Series({
                'mean_dist_mi': weighted_avg(g, dist_col, pop_col),
                'median_dist_mi': g[dist_col].median(),
                'travel_burden_share': g['travel_burden'].sum() / total_burden
            })
        ).reset_index()
        decile_summary['cum_travel_burden_share'] = decile_summary['travel_burden_share'].cumsum()
        return decile_summary

    for tech in ['all', 'ai', 'robo']:
        logging.info(f"\n--- Decile Table for '{tech.upper()}' Hospital Access ---")
        decile_table = create_decile_table(cbg_gdf, f'dist_{tech}', 'POPULATION')
        logging.info("\n" + decile_table.to_string(index=False))

    # --- 3. Absolute Gap Analysis (P90/P10) (User Request #2) ---
    logging.info("\n" + "-"*15 + " Absolute Gap Analysis (P90/P10) " + "-"*15)
    gap_data = []
    for tech, name in [('all', 'All Hospitals'), ('ai', 'AI'), ('robo', 'Robotics')]:
        dist_col = f'dist_{tech}'
        weighted_stats = wstats.DescrStatsW(cbg_gdf[dist_col], weights=cbg_gdf['POPULATION'])
        p10 = weighted_stats.quantile(0.10)
        p90 = weighted_stats.quantile(0.90)
        gap_data.append({
            'Access Type': name,
            'P10_dist_mi': p10,
            'P90_dist_mi': p90,
            'Absolute_Gap_mi': p90 - p10,
            'Ratio_Gap (P90/P10)': p90 / p10
        })
    gap_df = pd.DataFrame(gap_data)
    logging.info("Population-weighted distance percentiles and gaps:")
    logging.info("\n" + gap_df.to_string(index=False))

    # --- 4. Core Lorenz/Gini Plotting (Unchanged) ---
    logging.info("\n" + "-"*15 + " Lorenz Curve & Gini Coefficient Analysis " + "-"*15)

    def lorenz_xy(distance, weight):
        df = pd.DataFrame({'d': distance, 'w': weight}).sort_values('d')
        x = df['w'].cumsum() / df['w'].sum()
        y = (df['d'] * df['w']).cumsum() / (df['d'] * df['w']).sum()
        return pd.concat([pd.Series([0]), x]).values, pd.concat([pd.Series([0]), y]).values

    def calculate_gini(x, y):
        return 1 - 2 * trapz(y, x)

    x_ai, y_ai = lorenz_xy(cbg_gdf.dist_ai, cbg_gdf.POPULATION)
    gini_ai = calculate_gini(x_ai, y_ai)
    x_robo, y_robo = lorenz_xy(cbg_gdf.dist_robo, cbg_gdf.POPULATION)
    gini_robo = calculate_gini(x_robo, y_robo)
    x_all, y_all = lorenz_xy(cbg_gdf.dist_all, cbg_gdf.POPULATION)
    gini_all = calculate_gini(x_all, y_all)
    
    sns.set_style("whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(x_ai, y_ai, label=f"AI (Gini={gini_ai:.3f})", color='#0072B2', lw=2)
    ax1.plot(x_robo, y_robo, label=f"Robotics (Gini={gini_robo:.3f})", color='#D55E00', lw=2)
    ax1.plot(x_all, y_all, label=f"All Hospitals (Gini={gini_all:.3f})", color='#009E73', lw=2, linestyle=':')
    ax1.plot([0, 1], [0, 1], c='black', ls='--', label='Line of Perfect Equality')
    ax1.set_xlabel("Cumulative Share of Population", fontsize=12)
    ax1.set_ylabel("Cumulative Share of Travel Burden (Distance)", fontsize=12)
    ax1.set_title("Inequality of Access to Hospital Technology", fontsize=14, pad=15)
    ax1.legend()
    ax1.set_aspect('equal', 'box')

    gini_data = pd.DataFrame({
        'Technology': ['AI', 'Robotics', 'All Hospitals'],
        'Gini': [gini_ai, gini_robo, gini_all]
    })
    
    palette = {'AI': '#0072B2', 'Robotics': '#D55E00', 'All Hospitals': '#009E73'}
    sns.barplot(x='Technology', y='Gini', data=gini_data, ax=ax2, palette=palette, hue='Technology', legend=False)

    ax2.set_xlabel("Technology / Access Type", fontsize=12)
    ax2.set_ylabel("Gini Coefficient of Access", fontsize=12)
    ax2.set_title("Gini Coefficient", fontsize=14, pad=15)
    plt.suptitle("Geographic Inequality in Access to AI and Robotics in U.S. Hospitals", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    fig_path = FIG_DIR / "lorenz_gini_inequality_analysis_final.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved main Lorenz/Gini inequality plot to {fig_path}")
    plt.close(fig)

    # --- 5. Robustness Checks (CV, Epsilon-Shift Gini, Atkinson) ---
    logging.info("\n" + "-"*15 + " Robustness Checks " + "-"*15)

    logging.info("\n--- Coefficient of Variation (SD / Mean) ---")
    cv_data = {
        'Access Type': ['All Hospitals', 'AI', 'Robotics'],
        'CV': [
            cbg_gdf['dist_all'].std() / cbg_gdf['dist_all'].mean(),
            cbg_gdf['dist_ai'].std() / cbg_gdf['dist_ai'].mean(),
            cbg_gdf['dist_robo'].std() / cbg_gdf['dist_robo'].mean()
        ]
    }
    logging.info("\n" + pd.DataFrame(cv_data).to_string(index=False))

    logging.info("\n--- Gini Coefficient with 0.25-mile Epsilon-Shift ---")
    dist_shifted_all = cbg_gdf['dist_all'].clip(lower=0.25)
    x_s_all, y_s_all = lorenz_xy(dist_shifted_all, cbg_gdf.POPULATION)
    gini_shifted_all = calculate_gini(x_s_all, y_s_all)

    dist_shifted_ai = cbg_gdf['dist_ai'].clip(lower=0.25)
    x_s_ai, y_s_ai = lorenz_xy(dist_shifted_ai, cbg_gdf.POPULATION)
    gini_shifted_ai = calculate_gini(x_s_ai, y_s_ai)

    dist_shifted_robo = cbg_gdf['dist_robo'].clip(lower=0.25)
    x_s_robo, y_s_robo = lorenz_xy(dist_shifted_robo, cbg_gdf.POPULATION)
    gini_shifted_robo = calculate_gini(x_s_robo, y_s_robo)

    gini_shift_data = {
        'Access Type': ['All Hospitals', 'AI', 'Robotics'],
        'Original Gini': [gini_all, gini_ai, gini_robo],
        'Shifted Gini (0.25mi floor)': [gini_shifted_all, gini_shifted_ai, gini_shifted_robo]
    }
    logging.info("\n" + pd.DataFrame(gini_shift_data).to_string(index=False))

    logging.info("\n--- Atkinson Index (epsilon = 0.5) ---")
    def calculate_atkinson(values, weights, epsilon=0.5):
        """Calculates the Atkinson index for a given distribution."""
        if epsilon == 1:
            ratio = np.log(values)
            geo_mean = np.exp(np.average(ratio, weights=weights))
            arith_mean = np.average(values, weights=weights)
            return 1 - (geo_mean / arith_mean)
        else:
            weighted_mean = np.average(values, weights=weights)
            term = (values / weighted_mean)**(1 - epsilon)
            weighted_avg_term = np.average(term, weights=weights)
            return 1 - weighted_avg_term**(1 / (1 - epsilon))
    
    atkinson_all = calculate_atkinson(cbg_gdf['dist_all'], cbg_gdf['POPULATION'])
    atkinson_ai = calculate_atkinson(cbg_gdf['dist_ai'], cbg_gdf['POPULATION'])
    atkinson_robo = calculate_atkinson(cbg_gdf['dist_robo'], cbg_gdf['POPULATION'])
    
    atkinson_data = {
        'Access Type': ['All Hospitals', 'AI', 'Robotics'],
        'Atkinson Index (e=0.5)': [atkinson_all, atkinson_ai, atkinson_robo]
    }
    logging.info("\n" + pd.DataFrame(atkinson_data).to_string(index=False))

def model_adoption_drivers(gdf):
    """Models AI/Robotics adoption based on competition and hospital size."""
    logging.info("\n" + "="*20 + " MODELING ADOPTION DRIVERS " + "="*20)
    
    def run_logit(df, target_col, predictors):
        logging.info(f"--- Logistic Regression for: {target_col} ---")
        model_df = df[[target_col] + predictors].dropna()
        if model_df[target_col].nunique() < 2:
            logging.warning(f"Target '{target_col}' has only one class. Cannot build model.")
            return

        X = model_df[predictors].values
        y = model_df[target_col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(random_state=42, class_weight='balanced').fit(X_scaled, y)
        
        coeffs = pd.Series(model.coef_[0], index=predictors)
        logging.info(f"Predictors: {predictors}")
        logging.info(f"Model Accuracy: {model.score(X_scaled, y):.3f}")
        logging.info("Coefficients (log-odds):\n" + coeffs.to_string())
        
        logging.info("NOTE on 'k3_avg_miles' coefficient: A positive coefficient suggests that hospitals with greater distances to their nearest neighbors (i.e., less competition) are more likely to adopt the technology. A negative coefficient would have suggested a competitive driver.")
        
        logging.info("Classification Report:\n" + classification_report(y, model.predict(X_scaled), digits=3, zero_division=0))

    predictors = ['k3_avg_miles', 'bsc']
    run_logit(gdf, 'ai_flag', predictors)
    run_logit(gdf, 'robo_flag', predictors)

def perform_hotspot_analysis(hospital_gdf):
    """Performs Local Moran's I (LISA) analysis for adoption clusters."""
    logging.info("\n" + "="*20 + " HOT SPOT (LISA) ANALYSIS " + "="*20)
    if not PYSAL_AVAILABLE:
        logging.warning("PySAL not installed. Skipping hot spot analysis.")
        return
    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}. Skipping hot spot analysis.")
        return

    try:
        county_gdf = gpd.read_file(COUNTY_FILE)
        if 'GEOID' not in county_gdf.columns or 'STATEFP' not in county_gdf.columns:
            logging.error("County shapefile must have 'GEOID' and 'STATEFP' columns.")
            return

        hosp_proj = hospital_gdf.to_crs(county_gdf.crs)
        hosp_with_county = gpd.sjoin(hosp_proj, county_gdf[['GEOID', 'STATEFP', 'geometry']], how='left', predicate='within')
        county_agg = hosp_with_county.groupby('GEOID').agg(
            ai_count=('ai_flag', 'sum'),
            robo_count=('robo_flag', 'sum'),
            ai_intensity_sum=('ai_intensity', 'sum'),
            hospital_count=('hospital_id', 'count')
        ).reset_index()
        county_agg['ai_rate'] = (county_agg['ai_count'] / county_agg['hospital_count']).fillna(0)
        county_agg['robo_rate'] = (county_agg['robo_count'] / county_agg['hospital_count']).fillna(0)
        analysis_gdf_full = county_gdf.merge(county_agg, on='GEOID', how='left').fillna(0)
        
        analysis_gdf_full_proj = analysis_gdf_full.to_crs(AEA_CRS)

        def plot_lisa_map(gdf, lisa, title, filename):
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            lisa_cluster(lisa, gdf, p=0.05, ax=ax, legend_kwds={'loc': 'lower left'})
            ax.set_title(title, fontsize=16)
            ax.set_axis_off()
            if CONTEXTILY_AVAILABLE:
                cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logging.info(f"Saved LISA map to {filename}")
            plt.close(fig)

        for col in ['ai_rate', 'robo_rate', 'ai_intensity_sum']:
            logging.info(f"--- Running LISA for: {col} (Full US) ---")
            gdf = analysis_gdf_full_proj[analysis_gdf_full_proj.geometry.is_valid & ~analysis_gdf_full_proj.geometry.is_empty].copy()
            y = gdf[col].values
            w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
            lisa = esda.Moran_Local(y, w)
            title = f"Hot Spots of {col.replace('_', ' ').title()} by County (Full US)"
            filename = FIG_DIR / f"lisa_hotspot_{col}_full_us.png"
            plot_lisa_map(gdf, lisa, title, filename)

        logging.info("\n--- Creating focused hot spot map for Contiguous US ---")
        excluded_fips = ['02', '15', '72', '60', '66', '69', '78']
        
        gdf_contig_proj = analysis_gdf_full_proj[~analysis_gdf_full_proj['STATEFP'].isin(excluded_fips)].copy()
        gdf_contig_proj = gdf_contig_proj[gdf_contig_proj.geometry.is_valid & ~gdf_contig_proj.geometry.is_empty]

        column_to_plot = 'ai_intensity_sum'
        logging.info(f"--- Running LISA for: {column_to_plot} (Contiguous US) ---")
        y_contig = gdf_contig_proj[column_to_plot].values
        w_contig = libpysal.weights.Queen.from_dataframe(gdf_contig_proj, use_index=False, silence_warnings=True)
        lisa_contig = esda.Moran_Local(y_contig, w_contig)

        us_contig_basemap = analysis_gdf_full[~analysis_gdf_full['STATEFP'].isin(excluded_fips)]
        fig_contig, ax_contig = plt.subplots(1, 1, figsize=(12, 8))
        us_contig_basemap.plot(ax=ax_contig, color='#f0f0f0', edgecolor='white', linewidth=0.5)
        gdf_for_plot = gdf_contig_proj.to_crs(us_contig_basemap.crs)
        lisa_cluster(lisa_contig, gdf_for_plot, p=0.05, ax=ax_contig, legend_kwds={'loc': 'lower left'})
        
        ax_contig.set_title("Hot Spots of AI Adoption Intensity by County (Contiguous US)", fontsize=16)
        ax_contig.set_axis_off()
        ax_contig.set_xlim(-125, -66.5)
        ax_contig.set_ylim(24, 50)
        
        fig_path_contig = FIG_DIR / "lisa_hotspot_ai_intensity_sum_contiguous_us_clean.png"
        plt.savefig(fig_path_contig, dpi=300, bbox_inches='tight')
        logging.info(f"Saved CLEAN Contiguous US LISA cluster map to {fig_path_contig}")
        plt.close(fig_contig)

    except Exception as e:
        logging.error(f"Error during hot spot analysis: {e}", exc_info=True)

# ======================= K-NN GRAPH ANALYSIS (REFINED) =======================
def create_knn_graph(gdf, k):
    """
    Creates a k-Nearest Neighbors graph from a GeoDataFrame.
    It's critical that the input gdf is already filtered to the desired geography.

    Args:
        gdf (GeoDataFrame): A GeoDataFrame with point geometries.
        k (int): The number of nearest neighbors to find for each point.

    Returns:
        list: A list of tuples, where each tuple represents an edge
              connecting the integer-location (iloc) of two neighboring hospitals.
    """
    logging.info(f"\n" + "="*20 + f" CREATING K-NN GRAPH (k={k}) " + "="*20)
    if len(gdf) <= k:
        logging.warning(f"Number of hospitals ({len(gdf)}) is less than or equal to k ({k}). Cannot build graph.")
        return []

    # Reset index to ensure iloc-based results from BallTree are unambiguous
    gdf = gdf.reset_index(drop=True)

    coords_rad = np.deg2rad(gdf[['latitude', 'longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')

    # Query for k+1 neighbors because the first neighbor is the point itself
    # The `indices` array will contain the row's integer position for each neighbor
    _, indices = tree.query(coords_rad, k=k+1)

    # The indices array has shape (n_points, k+1)
    source_nodes = indices[:, 0]
    neighbor_nodes = indices[:, 1:]

    edges = []
    for i in range(len(source_nodes)):
        source_iloc = source_nodes[i]
        for j in range(k):
            neighbor_iloc = neighbor_nodes[i, j]
            edges.append((source_iloc, neighbor_iloc))

    logging.info(f"Successfully created k-NN graph with {len(gdf)} nodes and {len(edges)} edges.")
    return edges

def plot_knn_graph(gdf, edges, k, filename, title):
    """
    Generates a publication-quality visualization of the k-NN graph.
    (Simplified version assumes gdf and edges are already for the desired geography)

    Args:
        gdf (GeoDataFrame): The hospital data (pre-filtered for contiguous US).
        edges (list): The list of graph edges (from create_knn_graph).
        k (int): The 'k' value for titling.
        filename (str or Path): The path to save the output figure.
        title (str): The title for the plot.
    """
    logging.info(f"Generating visualization for the k={k} graph...")

    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}, skipping k-NN map.")
        return

    # Reset index of the plotting GDF to match the iloc-based edges
    gdf = gdf.reset_index(drop=True)

    # --- Prepare Basemap ---
    us_basemap = gpd.read_file(COUNTY_FILE)
    excluded_fips = ['02', '15', '72', '60', '66', '69', '78']
    us_contig_basemap = us_basemap[~us_basemap['STATEFP'].isin(excluded_fips)]

    # --- Create the Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # 1. Plot the basemap
    us_contig_basemap.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5)

    # 2. Plot the k-NN graph edges
    # The indices in `edges` now directly and correctly correspond to the rows in `gdf`
    for start_iloc, end_iloc in tqdm(edges, desc="Plotting graph edges"):
        start_geom = gdf.iloc[start_iloc].geometry
        end_geom = gdf.iloc[end_iloc].geometry
        ax.plot(
            [start_geom.x, end_geom.x],
            [start_geom.y, end_geom.y],
            color='darkgray',
            linewidth=0.5,
            alpha=0.6,
            zorder=1
        )

    # 3. Plot the hospital locations (nodes) on top
    gdf.plot(
        ax=ax,
        column='tech_type',
        categorical=True,
        legend=True,
        markersize=25,
        alpha=0.9,
        edgecolor='black',
        linewidth=0.5,
        zorder=2,
        legend_kwds={'title': "Technology Type", 'loc': 'lower left'}
    )

    # 4. Final Touches
    ax.set_title(title, fontsize=16)
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24, 50)
    ax.set_axis_off()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logging.info(f"Saved k-NN graph visualization to {filename}")
    plt.close(fig)

# ======================= YPLL ANALYSIS & VISUALIZATION =======================
def analyze_ypll_and_technology_access(engine, hospital_gdf, pop_gdf):
    """
    Performs a county-level multiple regression to test if access to technology
    predicts premature death (YPLL), controlling for composite SDOH indices,
    population, and census division.
    """
    logging.info("\n" + "="*20 + " YPLL REGRESSION ANALYSIS (V2) " + "="*20)
    
    # --- 1. Load County-Level Health and Socioeconomic Data (New Model) ---
    logging.info("Loading county-level data for new regression model.")
    query = """
    SELECT
        m.county_fips,
        m.social_economic_factors_score, -- IV4
        m.health_behaviors_score,        -- IV3
        m.physical_environment_score,    -- IV2
        m.population,                    -- ct1
        m.census_division,               -- ct2
        v.premature_death_raw_value      -- DV21 (YPLL)
    FROM vw_conceptual_model_adjpd m
    JOIN vw_conceptual_model_variables v ON m.county_fips = v.county_fips;
    """
    try:
        health_data = pd.read_sql(query, engine)
        health_data.rename(columns={'premature_death_raw_value': 'ypll'}, inplace=True)
        logging.info(f"Successfully loaded {len(health_data)} county records for new YPLL analysis.")
    except Exception as e:
        logging.error(f"FATAL: Could not load YPLL data for new model. Skipping analysis. Error: {e}")
        return

    # --- 2. Calculate County-Level Technology Access Metrics (Same as before) ---
    logging.info("Aggregating CBG-level distance metrics to county-level.")
    if 'GEOID' not in pop_gdf.columns:
        logging.error("FATAL: Population GeoJSON must contain a 'GEOID' column for county aggregation. Skipping.")
        return

    hospitals_ai = hospital_gdf[hospital_gdf['ai_flag'] == 1]
    hospitals_robo = hospital_gdf[hospital_gdf['robo_flag'] == 1]
    
    cbg_gdf_proj = pop_gdf.to_crs(AEA_CRS)
    cbg_centroids_proj = cbg_gdf_proj.geometry.centroid
    cbg_centroids_wgs84 = cbg_centroids_proj.to_crs(WGS84_CRS)
    cbg_coords_rad = np.deg2rad(cbg_centroids_wgs84.apply(lambda p: (p.y, p.x)).tolist())

    def get_nearest_distances(tech_gdf, cbg_coords, tech_name):
        tech_coords_rad = np.deg2rad(tech_gdf[['latitude', 'longitude']].values)
        tree = BallTree(tech_coords_rad, metric='haversine')
        distances_rad, _ = tree.query(cbg_coords, k=1)
        return distances_rad.flatten() * EARTH_RADIUS_MILES
    
    cbg_gdf = pop_gdf.copy()
    cbg_gdf['dist_ai'] = get_nearest_distances(hospitals_ai, cbg_coords_rad, 'AI')
    cbg_gdf['dist_robo'] = get_nearest_distances(hospitals_robo, cbg_coords_rad, 'Robotics')
    cbg_gdf['county_fips'] = cbg_gdf['GEOID'].str[:5]

    cbg_gdf['pop_dist_ai'] = cbg_gdf['dist_ai'] * cbg_gdf['POPULATION']
    cbg_gdf['pop_dist_robo'] = cbg_gdf['dist_robo'] * cbg_gdf['POPULATION']
    
    county_access = cbg_gdf.groupby('county_fips').agg(
        pop_dist_ai_sum=('pop_dist_ai', 'sum'),
        pop_dist_robo_sum=('pop_dist_robo', 'sum'),
        total_pop=('POPULATION', 'sum')
    ).reset_index()

    county_access['county_avg_dist_to_ai'] = county_access['pop_dist_ai_sum'] / county_access['total_pop']
    county_access['county_avg_dist_to_robo'] = county_access['pop_dist_robo_sum'] / county_access['total_pop']
    logging.info(f"Calculated population-weighted average access distances for {len(county_access)} counties.")
    
    # --- 3. Merge Datasets and Prepare for Regression ---
    final_df = pd.merge(health_data, county_access[['county_fips', 'county_avg_dist_to_ai', 'county_avg_dist_to_robo']], on='county_fips', how='inner')
    
    cols_to_standardize = [
        'county_avg_dist_to_ai', 'county_avg_dist_to_robo',
        'social_economic_factors_score', 'health_behaviors_score',
        'physical_environment_score', 'population'
    ]
    for col in cols_to_standardize + ['ypll']:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
    final_df.dropna(subset=['ypll', 'census_division'] + cols_to_standardize, inplace=True)
    
    df_for_simulation = final_df.copy()

    scaler = StandardScaler()
    final_df[cols_to_standardize] = scaler.fit_transform(final_df[cols_to_standardize])
    logging.info(f"Final dataset for regression has {len(final_df)} counties after cleaning.")
    
    ai_dist_idx = scaler.feature_names_in_.tolist().index('county_avg_dist_to_ai')
    ai_dist_sd = scaler.scale_[ai_dist_idx]
    logging.info(f"NOTE: For this model, one standard deviation in 'county_avg_dist_to_ai' is equal to {ai_dist_sd:.2f} miles.")

    # --- 4. Build and Run the OLS Regression Model ---
    formula = "ypll ~ " + " + ".join(cols_to_standardize) + " + C(census_division)"
    model = smf.ols(formula, data=final_df).fit(cov_type='HC1') # Use robust standard errors
    
    logging.info("--- OLS Regression Results (V2): Predicting YPLL ---")
    logging.info(f"\n{model.summary()}\n")
    
    # --- 5. Visualize Key Coefficients with Annotations ---
    logging.info("Generating annotated coefficient plot for key model variables.")
    
    coef_df = pd.DataFrame({
        'coef': model.params,
        'err': model.bse,
        'pvalue': model.pvalues
    })
    
    plot_vars = [v for v in coef_df.index if 'Intercept' not in v and 'census_division' not in v]
    coef_to_plot = coef_df.loc[plot_vars].copy()
    
    coef_to_plot['variable'] = [
        'Access: Avg. Dist. to AI', 'Access: Avg. Dist. to Robotics',
        'Socioeconomic Factors Score', 'Health Behaviors Score',
        'Physical Environment Score', 'Population'
    ]
    coef_to_plot['error_margin'] = coef_to_plot['err'] * 1.96

    fig, ax = plt.subplots(figsize=(12, 8))
    coef_to_plot.plot(kind='barh', x='variable', y='coef', xerr='error_margin',
                      ax=ax, legend=False, color=sns.color_palette("viridis", len(coef_to_plot)))
    
    ax.axvline(x=0, color='black', linestyle='--')
    
    for i, row in coef_to_plot.iterrows():
        p_val_text = f"p < 0.001" if row['pvalue'] < 0.001 else f"p = {row['pvalue']:.3f}"
        annotation = f"β = {row['coef']:.0f}\n{p_val_text}"
        text_x_pos = row['coef'] + row['error_margin'] + 50 if row['coef'] >= 0 else row['coef'] - row['error_margin'] - 50
        ha = 'left' if row['coef'] >= 0 else 'right'
        ax.text(text_x_pos, ax.get_yticks()[coef_to_plot.index.get_loc(i)], annotation, 
                va='center', ha=ha, fontsize=9)

    ax.set_title('Effect of Technology Access and SDOH on Premature Death (YPLL)', fontsize=16, pad=20)
    ax.set_xlabel('Change in YPLL (per 1 SD change in variable)', fontsize=12)
    ax.set_ylabel('')
    ax.set_xlim(ax.get_xlim()[0] * 1.3, ax.get_xlim()[1] * 1.3)
    plt.tight_layout()
    
    fig_path = FIG_DIR / "ypll_regression_coefficients_v2.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved annotated YPLL regression coefficient plot to {fig_path}")
    plt.close(fig)
    
    # --- 6. Run the 'What-If' Simulation & National Impact Estimate ---
    run_what_if_simulation(model, df_for_simulation, scaler)
    calculate_national_impact_estimate(model, df_for_simulation, scaler)

def run_what_if_simulation(model, df_unscaled, scaler):
    """
    Runs a 'what-if' simulation to estimate how much of the mortality gap
    between high- and low-access counties is explained by SDOH vs. technology.
    """
    logging.info("\n" + "="*20 + " 'WHAT-IF' SIMULATION " + "="*20)
    logging.info("Estimating the impact of SDOH on the mortality gap between high- and low-access counties.")

    desert_threshold = df_unscaled['county_avg_dist_to_ai'].quantile(0.90)
    oasis_threshold = df_unscaled['county_avg_dist_to_ai'].quantile(0.10)
    deserts_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] >= desert_threshold]
    oases_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] <= oasis_threshold]

    logging.info(f"Identified {len(deserts_df)} 'Technology Desert' counties (worst 10% access).")
    logging.info(f"Identified {len(oases_df)} 'Technology Oasis' counties (best 10% access).")

    ypll_desert_actual = deserts_df['ypll'].mean()
    ypll_oasis_actual = oases_df['ypll'].mean()
    observed_mortality_gap = ypll_desert_actual - ypll_oasis_actual

    logging.info(f"\n[STEP 1] Observed Data:")
    logging.info(f"  - Avg. YPLL in Technology Deserts: {ypll_desert_actual:,.0f}")
    logging.info(f"  - Avg. YPLL in Technology Oases:  {ypll_oasis_actual:,.0f}")
    logging.info(f"  - Total Observed Mortality Gap:   {observed_mortality_gap:,.0f} years of potential life lost")

    model_predictors = [v for v in model.params.index if 'Intercept' not in v and 'C(census_division)' not in v]
    typical_desert_profile = deserts_df[model_predictors].mean().to_frame().T
    hypothetical_profile = typical_desert_profile.copy()
    hypothetical_profile['county_avg_dist_to_ai'] = oases_df['county_avg_dist_to_ai'].mean()
    
    logging.info(f"\n[STEP 2] Hypothetical Scenario:")
    logging.info("  - Simulating a county with 'Desert-level' socioeconomic factors but 'Oasis-level' AI access.")

    cols_to_scale = scaler.feature_names_in_
    hypothetical_profile_scaled = scaler.transform(hypothetical_profile[cols_to_scale])
    hypothetical_df_scaled = pd.DataFrame(hypothetical_profile_scaled, columns=cols_to_scale)
    hypothetical_df_scaled['census_division'] = df_unscaled['census_division'].mode()[0]
    predicted_ypll = model.predict(hypothetical_df_scaled)[0]

    ypll_reduction_from_sdoh = ypll_desert_actual - predicted_ypll
    pct_gap_explained_by_sdoh = (ypll_reduction_from_sdoh / observed_mortality_gap) * 100 if observed_mortality_gap > 0 else 0

    logging.info(f"\n[STEP 3] Simulation Results:")
    logging.info(f"  - The actual YPLL in a typical desert county is {ypll_desert_actual:,.0f}.")
    logging.info(f"  - The model predicts that if this county got best-in-class AI access, its YPLL would only fall to {predicted_ypll:,.0f}.")
    logging.info(f"  - This implies that {ypll_reduction_from_sdoh:,.0f} of the {observed_mortality_gap:,.0f} mortality gap is associated with the bundle of socioeconomic factors.")
    logging.info(f"\n[CONCLUSION] >> Approximately {pct_gap_explained_by_sdoh:.1f}% of the mortality gap between the best- and worst-access counties is associated with foundational socioeconomic and health factors, not technology access itself.")
    logging.info("="*52)

def calculate_national_impact_estimate(model, df_unscaled, scaler):
    """
    Estimates the total national reduction in YPLL and equivalent lives saved if all
    non-oasis counties were given best-in-class AI technology access.
    """
    logging.info("\n" + "="*20 + " NATIONAL IMPACT ESTIMATE " + "="*20)

    oasis_threshold = df_unscaled['county_avg_dist_to_ai'].quantile(0.10)
    oases_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] <= oasis_threshold]
    non_oases_df = df_unscaled[df_unscaled['county_avg_dist_to_ai'] > oasis_threshold].copy()
    target_ai_access = oases_df['county_avg_dist_to_ai'].mean()

    logging.info(f"Estimating national impact by improving AI access for {len(non_oases_df)} counties.")
    logging.info(f"Target 'Best-in-Class' AI Access (avg. of best 10%): {target_ai_access:.2f} miles.")

    cols_to_scale = scaler.feature_names_in_
    
    non_oases_df_scaled = pd.DataFrame(scaler.transform(non_oases_df[cols_to_scale]), columns=cols_to_scale, index=non_oases_df.index)
    non_oases_df_scaled['census_division'] = non_oases_df['census_division']
    predicted_ypll_before = model.predict(non_oases_df_scaled)

    hypothetical_df = non_oases_df.copy()
    hypothetical_df['county_avg_dist_to_ai'] = target_ai_access
    hypothetical_df_scaled = pd.DataFrame(scaler.transform(hypothetical_df[cols_to_scale]), columns=cols_to_scale, index=hypothetical_df.index)
    hypothetical_df_scaled['census_division'] = hypothetical_df['census_division']
    predicted_ypll_after = model.predict(hypothetical_df_scaled)
    
    ypll_reduction_per_county = predicted_ypll_before - predicted_ypll_after
    total_national_ypll_reduction = np.sum(ypll_reduction_per_county)

    YPLL_PER_DEATH = 25
    equivalent_lives_saved = total_national_ypll_reduction / YPLL_PER_DEATH
    
    logging.info("\n[NATIONAL IMPACT ESTIMATE] Results:")
    logging.info(f"  - Total potential reduction in YPLL across all improved counties: {total_national_ypll_reduction:,.0f}")
    logging.info(f"  - Assuming {YPLL_PER_DEATH} years of life lost per preventable death, this is equivalent to approximately {equivalent_lives_saved:,.0f} fewer deaths nationally.")
    logging.info("="*56)

def generate_summary_visualizations(gdf):
    """Creates general-purpose visualizations for the final report."""
    logging.info("\n" + "="*20 + " GENERATING SUMMARY VISUALIZATIONS " + "="*20)
    sns.set_theme(style="whitegrid")

    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}, skipping all map visualizations.")
    else:
        try:
            logging.info("Preparing basemap and data for map visualizations...")
            us_basemap = gpd.read_file(COUNTY_FILE)
            excluded_fips = ['02', '15', '72', '60', '66', '69', '78']
            us_contig_basemap = us_basemap[~us_basemap['STATEFP'].isin(excluded_fips)]

            excluded_states = ['AK', 'HI', 'PR']
            gdf_contig_points = gdf[~gdf['state'].isin(excluded_states)].copy()

            logging.info("Generating map 1: All hospital tech types (Contiguous US).")
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            us_contig_basemap.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5)
            gdf_contig_points.plot(
                ax=ax, column='tech_type', categorical=True, legend=True, markersize=20,
                alpha=0.8, legend_kwds={'title': "Technology Type", 'loc': 'lower left'}
            )
            ax.set_title('Distribution of US Hospitals by AI and Robotics Adoption (Contiguous US)', fontsize=16)
            ax.set_xlim(-125, -66.5); ax.set_ylim(24, 50); ax.set_axis_off()
            fig_path = FIG_DIR / "hospital_distribution_contiguous_US.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved contiguous US map to {fig_path}")
            plt.close(fig)

        except Exception as e:
            logging.error(f"Could not generate distribution map(s): {e}", exc_info=True)

    sns.set_theme(style="whitegrid", palette="viridis")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=gdf, x='nearest_miles', bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Distances to Nearest Hospital (Cleaned Data)')
    ax.set_xlabel('Distance to Nearest Hospital (miles)'); ax.set_ylabel('Number of Hospitals')
    ax.axvline(gdf['nearest_miles'].median(), color='red', linestyle='--', label=f"Median: {gdf['nearest_miles'].median():.1f} mi")
    ax.legend()
    fig_path = FIG_DIR / "nearest_distance_histogram.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved distance histogram to {fig_path}")
    plt.close(fig)

# ======================= MAIN EXECUTION ==========================
def main():
    """Main function to orchestrate the analysis workflow."""
    setup_environment()
    engine = connect_to_db()

    # --- Load Data ---
    hospital_df = load_hospital_data(engine)
    pop_gdf = load_population_data(POP_FILE)

    # --- Preprocess and Clean ---
    hospital_gdf_preprocessed = preprocess_data(hospital_df)
    hospital_gdf_cleaned = calculate_proximity_metrics(hospital_gdf_preprocessed)

    # --- Run Core Analyses ---
    analyze_population_coverage(hospital_gdf_cleaned, pop_gdf)
    analyze_inequality_with_lorenz(hospital_gdf_cleaned, pop_gdf)
    analyze_ypll_and_technology_access(engine, hospital_gdf_cleaned, pop_gdf)
    model_adoption_drivers(hospital_gdf_cleaned)
    perform_hotspot_analysis(hospital_gdf_cleaned)

    # --- NEW: Run and Visualize k-NN Graph Analysis (CORRECTED LOGIC) ---
    logging.info("\n--- Preparing data for Contiguous US k-NN Graph ---")
    excluded_states = ['AK', 'HI', 'PR', 'GU', 'AS', 'VI', 'MP']
    hospital_gdf_contig = hospital_gdf_cleaned[~hospital_gdf_cleaned['state'].isin(excluded_states)].copy()
    logging.info(f"Filtered to {len(hospital_gdf_contig)} hospitals in the contiguous US for graph analysis.")

    K_VALUE = 5
    # Build the graph ONLY with contiguous US data to ensure all edges are internal
    knn_edges = create_knn_graph(hospital_gdf_contig, k=K_VALUE)
    
    if knn_edges: # Only plot if the graph was created successfully
        plot_knn_graph(
            gdf=hospital_gdf_contig, # Pass the filtered GDF
            edges=knn_edges,
            k=K_VALUE,
            filename=FIG_DIR / f"knn_graph_k{K_VALUE}_contiguous_us_corrected.png",
            title=f'k-Nearest Neighbor Graph of U.S. Hospitals (k={K_VALUE}, Contiguous US)'
        )

    # --- Existing final steps ---
    generate_summary_visualizations(hospital_gdf_cleaned)

    # --- Save Final Output ---
    try:
        output_path = BASE_DIR / "hospital_ai_robotics_enriched.parquet"
        final_df = pd.DataFrame(hospital_gdf_cleaned.drop(columns='geometry'))
        final_df['geometry_wkt'] = hospital_gdf_cleaned.geometry.to_wkt()
        final_df.to_parquet(output_path, index=False)
        logging.info(f"\nSuccessfully saved enriched data to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save output file: {e}")

    logging.info("=" * 60)
    logging.info("RUN COMPLETED")
    logging.info("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An unhandled exception occurred during the main execution.")
        sys.exit(1)

# ======================= ADVANCED SUGGESTION FOR FURTHER ANALYSIS =======================
# For a deeper analysis, you can convert your edge list into a formal graph object
# using the networkx library. This would allow you to calculate network metrics
# for your dissertation, such as centrality, connected components, etc.

# Example:
#
# import networkx as nx
#
# # In your main function, after creating your edges:
# # G = nx.Graph()
# # G.add_nodes_from(range(len(hospital_gdf_cleaned))) # Add all hospitals as nodes
# # G.add_edges_from(knn_edges)
#
# # Now you can calculate metrics
# # For example, find the largest connected component
# # largest_cc = max(nx.connected_components(G), key=len)
# # print(f"The largest connected network of hospitals contains {len(largest_cc)} nodes.")
#
# # Or calculate degree centrality (how many neighbors each hospital has)
# # degree_centrality = nx.degree_centrality(G)
# # You could add this back to your GeoDataFrame
# # hospital_gdf_cleaned['degree_centrality'] = hospital_gdf_cleaned.index.map(degree_centrality)