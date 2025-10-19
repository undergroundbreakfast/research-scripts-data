#!/usr/bin/env python3
# ==============================================================================
#  geospatial_hospital_ai_robotics_analysis.py
#  Author:  Aaron Johnson
#  Last updated: 2025-07-19
#
#  VERSION 18.0 - Adds detailed county-level analysis. Calculates population-
#                 weighted average driving times to the nearest AI-enabled
#                 hospital and the nearest Robotics-enabled hospital for
#                 every US county.
#
#  VERSION 17.1 - Corrects a data joining error in the AHEI analysis function.
#                 The script now reliably derives state FIPS codes from the
#                 county FIPS column instead of relying on state name matching,
#                 ensuring the AHEI calculations proceed correctly.
#
#  VERSION 17.0 - Integrates state-level Gini coefficient calculation to
#                 generate the AI Hospital Equity Index (AHEI).
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
from numpy import trapz

# --- Visualization Libraries ---
import matplotlib
matplotlib.use('Cairo') # Use Cairo backend for high-quality, reliable output
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Optional Geospatial Libraries ---
try:
    from matplotlib_scalebar.scalebar import ScaleBar
    SCALEBAR_AVAILABLE = True
except ImportError:
    SCALEBAR_AVAILABLE = False
    print("Warning: 'matplotlib-scalebar' not found. Scale bar will not be added to poster. Run: pip install matplotlib-scalebar")

try:
    import contextily as cx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    print("Warning: 'contextily' not found. Basemaps will not be added to maps.")

# ======================= CONFIGURATION ==========================
BASE_DIR = Path(__file__).resolve().parent
SHAPE_DIR = BASE_DIR / "shapefiles"
FIG_DIR = BASE_DIR / "figures"
LOG_FILE = BASE_DIR / "analysis_log.txt"
CX_CACHE = BASE_DIR / "tile_cache"

COUNTY_FILE = SHAPE_DIR / "tl_2024_us_county.shp"
STATES_FILE = SHAPE_DIR / "tl_2023_us_state.shp"
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

def generate_poster_map(hospital_gdf, pop_gdf):
    """
    Generates a 36" x 48" banner map showing which Census Block Groups are
    within a specified driving time of an AI-enabled hospital.
    """
    logging.info("\n" + "="*20 + " GENERATING 36x48 ACCESSIBILITY POSTER MAP " + "="*20)
    if pop_gdf is None:
        logging.error("Population data not available. Cannot generate poster map.")
        return

    logging.info("Step 1/5: Preparing data for the accessibility map...")
    excluded_states = ['AK', 'HI', 'PR', 'GU', 'AS', 'VI', 'MP']
    hospitals_ai = hospital_gdf[(hospital_gdf['ai_flag'] == 1) & (~hospital_gdf['state'].isin(excluded_states))].copy()
    if hospitals_ai.empty:
        logging.warning("No AI-enabled hospitals in contiguous US. Cannot generate accessibility map.")
        return
    if 'STATEFP' not in pop_gdf.columns:
        pop_gdf['STATEFP'] = pop_gdf['GEOID'].str[:2]
    excluded_fips = ['02', '15', '72', '60', '66', '69', '78']
    pop_contig = pop_gdf[~pop_gdf['STATEFP'].isin(excluded_fips)].copy()
    logging.info(f"Filtered to {len(hospitals_ai)} AI hospitals and {len(pop_contig)} block groups.")

    logging.info(f"Step 2/5: Calculating distance and classifying based on {DRIVE_MINUTES}-minute threshold...")
    cbg_centroids_wgs84 = pop_contig.to_crs(AEA_CRS).geometry.centroid.to_crs(WGS84_CRS)
    cbg_coords_rad = np.deg2rad(cbg_centroids_wgs84.apply(lambda p: (p.y, p.x)).tolist())
    ai_coords_rad = np.deg2rad(hospitals_ai[['latitude', 'longitude']].values)
    tree = BallTree(ai_coords_rad, metric='haversine')
    distances_rad, _ = tree.query(cbg_coords_rad, k=1)
    distances_miles = distances_rad.flatten() * EARTH_RADIUS_MILES
    pop_contig['drive_time_min'] = (distances_miles / AVG_SPEED_MPH) * 60

    cat_within = f'<= {DRIVE_MINUTES} Minute Drive'
    cat_outside = f'> {DRIVE_MINUTES} Minute Drive'
    pop_contig['access_category'] = np.where(
        pop_contig['drive_time_min'] <= DRIVE_MINUTES, cat_within, cat_outside
    )
    pop_contig['access_category'] = pd.Categorical(
        pop_contig['access_category'], categories=[cat_within, cat_outside], ordered=True
    )
    logging.info("Classification complete.")

    logging.info("Step 3/5: Creating poster canvas and rendering classified block group layer...")
    POSTER_W_IN = 48.0
    POSTER_H_IN = 36.0
    POSTER_DPI = 150
    pop_contig_proj = pop_contig.to_crs(AEA_CRS)

    fig, ax = plt.subplots(1, 1, figsize=(POSTER_W_IN, POSTER_H_IN))
    ax.set_aspect('equal')
    plot_colors = ['blue', 'grey']
    pop_contig_proj.plot(
        ax=ax, column='access_category', categorical=True, legend=True,
        cmap=mcolors.ListedColormap(plot_colors), linewidth=0, rasterized=True,
        legend_kwds={
            'title': 'Accessibility Status', 'loc': 'lower right',
            'bbox_to_anchor': (1, 0.08), 'fontsize': 24, 'title_fontsize': 28,
            'frameon': True, 'edgecolor': 'black'
        }
    )
    logging.info("Rasterized layer has been plotted.")

    logging.info("Step 4/5: Overlaying vector-based state outlines and map furniture...")
    if STATES_FILE.exists():
        states_proj = gpd.read_file(STATES_FILE).to_crs(AEA_CRS)
        states_contig_proj = states_proj[~states_proj['STUSPS'].isin(excluded_states)]
        states_contig_proj.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=0.75, alpha=0.8)

    if SCALEBAR_AVAILABLE:
        try:
            ax.add_artist(ScaleBar(
                dx=1, units='m', fixed_value=320, fixed_units='km',
                location='lower left', length_fraction=0.25, box_alpha=0.8,
                font_properties={'size': 18}, scale_loc='bottom', label=f"~200 miles"
            ))
            logging.info("Successfully added scale bar (320 km ≈ 200 miles).")
        except Exception as e:
            logging.warning(f"Scale bar creation failed: {e}. Adding text-based scale.")
            ax.text(0.02, 0.02, "Scale: Approx. 200 miles", ha='left', va='bottom', transform=ax.transAxes,
                    fontsize=18, bbox=dict(facecolor='white', alpha=0.8, pad=5))

    ax.set_axis_off()
    ax.set_title(
        f"{DRIVE_MINUTES}-Minute Driving Access to AI-Enabled Hospitals in the Contiguous U.S.",
        fontsize=48, pad=30, fontweight='bold'
    )
    ax.text(0.5, 0.96,
            f"Blue areas represent Census Block Groups within a {DRIVE_MINUTES}-minute drive of an AI-enabled hospital",
            ha='center', va='center', transform=ax.transAxes, fontsize=28)
    ax.text(0.99, 0.01,
            f"Author: Aaron Johnson | Data: AHA (2024), US Census (2020)\nMap Projection: USA Contiguous Albers Equal Area Conic (EPSG:{AEA_CRS})",
            ha='right', va='bottom', transform=ax.transAxes, fontsize=14, color='gray')
    logging.info("Map furniture (title, legend, etc.) added.")

    logging.info("Step 5/5: Saving the final poster to PDF... This may take a moment.")
    output_filename = BASE_DIR / f"poster_map_accessibility_{DRIVE_MINUTES}min_final.pdf"
    plt.savefig(
        output_filename, dpi=POSTER_DPI, format='pdf', bbox_inches='tight', pad_inches=0.5
    )
    logging.info(f"\nSUCCESS: Poster map saved to:\n{output_filename}")
    plt.close(fig)

### --- GINI HELPER FUNCTION (Unchanged) --- ###
def lorenz_gini(distance, weight):
    """
    Helper to calculate the Gini coefficient for a given distribution,
    representing inequality of travel burden.
    """
    distance = pd.Series(distance)
    weight = pd.Series(weight)
    df = pd.DataFrame({'d': distance, 'w': weight}).sort_values('d')
    total_burden = (df['d'] * df['w']).sum()
    if total_burden == 0:
        return 0.0
    cum_pop = df['w'].cumsum() / df['w'].sum()
    cum_burden = (df['d'] * df['w']).cumsum() / total_burden
    x = pd.concat([pd.Series([0]), cum_pop]).values
    y = pd.concat([pd.Series([0]), cum_burden]).values
    gini = 1 - 2 * np.trapz(y, x)
    return gini

### --- CORRECTED FUNCTION FOR STATE-LEVEL ANALYSIS AND AHEI --- ###
def analyze_state_level_access(hospital_gdf, pop_gdf, states_filepath, output_dir):
    """
    Calculates state-level statistics, including the AI Hospital Equity Index (AHEI),
    which combines population coverage, access equity (Gini), travel time, and
    adoption intensity.
    """
    logging.info("\n" + "="*20 + " STATE-LEVEL ACCESSIBILITY & AHEI ANALYSIS " + "="*20)

    # --- 1. Input Validation ---
    if pop_gdf is None or pop_gdf.empty:
        logging.error("Population data is not available. Skipping state-level analysis.")
        return
    if not states_filepath.exists():
        logging.error(f"States shapefile not found at {states_filepath}. Skipping state-level analysis.")
        return

    # --- 2. Data Preparation ---
    logging.info("Step 1/7: Preparing data for state-level analysis...")
    states_gdf = gpd.read_file(states_filepath).to_crs(AEA_CRS)
    states_gdf['state_total_area_sqkm'] = states_gdf.geometry.area / 1_000_000

    # === FIX START: Use robust county_fips to derive state FIPS ===
    if 'county_fips' not in hospital_gdf.columns or hospital_gdf['county_fips'].isnull().all():
        logging.error("FATAL: 'county_fips' column is missing or empty in hospital data. Cannot proceed.")
        return
    
    hospital_gdf['county_fips_str'] = hospital_gdf['county_fips'].astype(str).str.split('.').str[0].str.zfill(5)
    hospital_gdf['STATEFP'] = hospital_gdf['county_fips_str'].str[:2]
    
    hospitals_ai = hospital_gdf[hospital_gdf['ai_flag'] == 1].copy()
    # === FIX END ===
    
    if hospitals_ai.empty:
        logging.warning("No AI-enabled hospitals found in the dataset. Cannot perform state-level analysis.")
        return
    
    logging.info(f"Found {len(hospitals_ai)} AI-enabled hospitals with valid state FIPS for analysis.")

    if 'STATEFP' not in pop_gdf.columns:
        if 'GEOID' in pop_gdf.columns:
            pop_gdf['STATEFP'] = pop_gdf['GEOID'].str[:2]
        else:
            logging.error("Cannot determine state for population data (no 'GEOID' or 'STATEFP').")
            return
    pop_proj = pop_gdf.to_crs(AEA_CRS)

    # --- 3. Population and Area Coverage ---
    logging.info(f"Step 2/7: Creating {DRIVE_MINUTES}-minute drive buffer around {len(hospitals_ai)} AI hospitals...")
    hospitals_ai_proj = hospitals_ai.to_crs(AEA_CRS)
    radius_meters = RADIUS_MILES * 1609.344
    coverage_union = hospitals_ai_proj.buffer(radius_meters).unary_union
    coverage_gdf = gpd.GeoDataFrame(geometry=[coverage_union], crs=AEA_CRS)

    logging.info("Step 3/7: Calculating land area and population coverage per state...")
    try:
        state_coverage_intersect = gpd.overlay(states_gdf, coverage_gdf, how='intersection', keep_geom_type=False)
        state_coverage_intersect['covered_area_sqkm'] = state_coverage_intersect.geometry.area / 1_000_000
        area_results = state_coverage_intersect.groupby('STATEFP')['covered_area_sqkm'].sum().reset_index()
    except Exception as e:
        logging.error(f"Could not calculate area coverage due to a geospatial error: {e}")
        area_results = pd.DataFrame(columns=['STATEFP', 'covered_area_sqkm'])

    pop_covered = gpd.sjoin(pop_proj, coverage_gdf, how='inner', predicate='intersects')
    total_pop_by_state = pop_proj.groupby('STATEFP')['POPULATION'].sum()
    covered_pop_by_state = pop_covered.groupby('STATEFP')['POPULATION'].sum()
    pop_results = pd.DataFrame({
        'total_population': total_pop_by_state,
        'population_in_30min_drive': covered_pop_by_state
    }).reset_index()
    pop_results['population_in_30min_drive'].fillna(0, inplace=True)
    pop_results['population_in_30min_drive'] = pop_results['population_in_30min_drive'].astype(int)

    # --- 4. AHEI Component Calculation ---
    logging.info("Step 4/7: Calculating components for the AI Hospital Equity Index (AHEI)...")
    
    logging.info("...calculating distance from each of ~240k CBGs to nearest AI hospital. This may take a minute.")
    cbg_centroids_wgs84 = pop_proj.geometry.centroid.to_crs(WGS84_CRS)
    cbg_coords_rad = np.deg2rad(cbg_centroids_wgs84.apply(lambda p: (p.y, p.x)).tolist())
    ai_coords_rad = np.deg2rad(hospitals_ai[['latitude', 'longitude']].values)
    tree = BallTree(ai_coords_rad, metric='haversine')
    distances_rad, _ = tree.query(cbg_coords_rad, k=1)
    pop_proj['dist_ai_miles'] = distances_rad.flatten() * EARTH_RADIUS_MILES

    logging.info("...calculating Access Gini and average travel times by state.")
    with tqdm(total=pop_proj['STATEFP'].nunique(), desc="Calculating State Ginis") as pbar:
        state_ginis = (
            pop_proj.groupby('STATEFP').apply(lambda g: (pbar.update(1), lorenz_gini(g['dist_ai_miles'], g['POPULATION']))[1])
            .rename('access_gini').reset_index()
        )
    
    pop_proj['time_ai_min'] = (pop_proj['dist_ai_miles'] / AVG_SPEED_MPH) * 60
    state_avg_time = (
        pop_proj.groupby('STATEFP').apply(lambda g: np.average(g['time_ai_min'], weights=g['POPULATION']))
        .rename('avg_travel_time_min').reset_index()
    )

    logging.info("...calculating adoption intensity by state.")
    ai_hospitals_by_state = hospitals_ai.groupby('STATEFP').size().rename('ai_hospital_count')
    state_intensity = pd.merge(ai_hospitals_by_state, pop_results[['STATEFP', 'total_population']], on='STATEFP', how='right').fillna(0)
    state_intensity['ai_hospitals_per_million'] = (state_intensity['ai_hospital_count'] / state_intensity['total_population'].replace(0, np.nan)) * 1_000_000

    # --- 5. Merge and Finalize Metrics ---
    logging.info("Step 5/7: Merging all state-level metrics...")
    final_df = states_gdf[['STATEFP', 'STUSPS', 'NAME', 'state_total_area_sqkm']].copy()
    final_df = pd.merge(final_df, pop_results, on='STATEFP', how='left')
    final_df = pd.merge(final_df, area_results, on='STATEFP', how='left')
    final_df = pd.merge(final_df, state_ginis, on='STATEFP', how='left')
    final_df = pd.merge(final_df, state_avg_time, on='STATEFP', how='left')
    final_df = pd.merge(final_df, state_intensity[['STATEFP', 'ai_hospital_count', 'ai_hospitals_per_million']], on='STATEFP', how='left')
    final_df.fillna(0, inplace=True)

    # --- 6. Calculate AHEI Scores ---
    logging.info("Step 6/7: Normalizing components and calculating final AHEI score...")

    final_df['pct_pop_in_30min_drive'] = (final_df['population_in_30min_drive'] / final_df['total_population'].replace(0, np.nan)) * 100
    final_df['PopCoverageScore'] = final_df['pct_pop_in_30min_drive'].fillna(0)

    GMIN, GMAX = 0.30, 0.65
    final_df['AccessEquityScore'] = 100 * (GMAX - final_df['access_gini']) / (GMAX - GMIN)
    final_df['AccessEquityScore'] = final_df['AccessEquityScore'].clip(0, 100)

    TT_MIN, TT_MAX = 10, 75
    final_df['TravelTimeScore'] = 100 * (TT_MAX - final_df['avg_travel_time_min']) / (TT_MAX - TT_MIN)
    final_df['TravelTimeScore'] = final_df['TravelTimeScore'].clip(0, 100)
    
    max_intensity = final_df['ai_hospitals_per_million'].quantile(0.98)
    if max_intensity == 0: max_intensity = 1
    final_df['AdoptionIntensityScore'] = 100 * (final_df['ai_hospitals_per_million'] / max_intensity)
    final_df['AdoptionIntensityScore'] = final_df['AdoptionIntensityScore'].clip(0, 100)
    
    final_df['AHEI'] = (
        0.50 * final_df['PopCoverageScore'] +
        0.20 * final_df['AccessEquityScore'] +
        0.15 * final_df['TravelTimeScore'] +
        0.15 * final_df['AdoptionIntensityScore']
    ).round(2)
    
    final_df['pct_area_in_30min_drive'] = (final_df['covered_area_sqkm'] / final_df['state_total_area_sqkm'].replace(0, np.nan)) * 100
    final_df.fillna(0, inplace=True)
    final_df.rename(columns={'STUSPS': 'state_abbr', 'NAME': 'state_name'}, inplace=True)
    
    output_cols = [
        'state_name', 'state_abbr', 'AHEI', 'access_gini',
        'PopCoverageScore', 'AccessEquityScore', 'TravelTimeScore', 'AdoptionIntensityScore',
        'avg_travel_time_min', 'ai_hospitals_per_million', 'ai_hospital_count',
        'population_in_30min_drive', 'total_population', 'pct_pop_in_30min_drive',
        'covered_area_sqkm', 'state_total_area_sqkm', 'pct_area_in_30min_drive'
    ]
    final_df = final_df.reindex(columns=output_cols).sort_values(by='AHEI', ascending=False).reset_index(drop=True)
    final_df = final_df[final_df['total_population'] > 0]

    # --- 7. Output Results ---
    logging.info("Step 7/7: Displaying results and saving to CSV...")
    print("\n\n" + "="*90)
    print("      AI Hospital Equity Index (AHEI) by State")
    print(f"      (Based on a {DRIVE_MINUTES}-minute drive at {AVG_SPEED_MPH} MPH)")
    print("="*90)
    
    display_df = final_df.copy()
    display_df['AHEI'] = display_df['AHEI'].map('{:.1f}'.format)
    display_df['Access Gini'] = display_df['access_gini'].map('{:.3f}'.format)
    display_df['Pop. Coverage'] = display_df['PopCoverageScore'].map('{:.1f}%'.format)
    display_df['Avg. Travel Time (min)'] = display_df['avg_travel_time_min'].map('{:.1f}'.format)

    print(display_df[[
        'state_name', 'AHEI', 'Pop. Coverage', 'Access Gini', 'Avg. Travel Time (min)'
    ]].to_string(index=False))
    print("="*90)

    csv_path = output_dir / "state_level_ai_hospital_equity_index.csv"
    final_df.to_csv(csv_path, index=False, float_format='%.4f')
    logging.info(f"Successfully saved detailed state-level AHEI analysis to: {csv_path}")

# ======================= NEWLY ADDED COUNTY ANALYSIS FUNCTIONS ==========================
def _calculate_weighted_drive_time_by_county(pop_gdf_with_coords, target_hospitals_gdf, avg_speed_mph, earth_radius_miles):
    """
    Internal helper function to calculate population-weighted drive times.
    This function expects a pop_gdf that already has a 'coords_rad' column.

    Args:
        pop_gdf_with_coords (GeoDataFrame): Population data with pre-calculated radian coordinates.
        target_hospitals_gdf (GeoDataFrame): The filtered set of hospitals to calculate distance to.
        avg_speed_mph (float): The average speed for travel time conversion.
        earth_radius_miles (float): Earth's radius for haversine calculation.

    Returns:
        pandas.Series: A Series with COUNTYFP as the index and the calculated
                       population-weighted average drive time as the value.
    """
    if target_hospitals_gdf.empty:
        return None # Return None if no hospitals match the criteria

    # Prepare coordinates and build the BallTree for the target hospitals
    hospital_coords_rad = np.deg2rad(target_hospitals_gdf[['latitude', 'longitude']].values)
    tree = BallTree(hospital_coords_rad, metric='haversine')
    
    # Query the tree to find the nearest hospital for each CBG centroid
    # pop_gdf_with_coords['coords_rad'] is a pre-calculated list of radian coordinate tuples
    distances_rad, _ = tree.query(pop_gdf_with_coords['coords_rad'].tolist(), k=1)
    
    # Create a temporary DataFrame to hold the CBG-level results
    temp_pop_df = pop_gdf_with_coords.copy()
    temp_pop_df['dist_miles'] = distances_rad.flatten() * earth_radius_miles
    temp_pop_df['drive_time_min'] = (temp_pop_df['dist_miles'] / avg_speed_mph) * 60
    
    # Define the weighted average function for the apply step
    def weighted_avg(g):
        return np.average(g['drive_time_min'], weights=g['POPULATION'])

    # Group by county and apply the weighted average calculation
    county_avg_time = temp_pop_df.groupby('COUNTYFP').apply(weighted_avg)
    
    return county_avg_time

def analyze_county_level_access_detailed(hospital_gdf, pop_gdf, county_filepath, output_dir):
    """
    Calculates population-weighted average driving times to the nearest AI-enabled
    and Robotics-enabled hospitals for every county in the US.
    """
    logging.info("\n" + "="*20 + " DETAILED COUNTY-LEVEL ACCESS ANALYSIS (AI & ROBOTICS) " + "="*20)

    # --- 1. Input Validation ---
    if pop_gdf is None or pop_gdf.empty:
        logging.error("Population data is not available. Skipping county-level analysis.")
        return
    if not county_filepath.exists():
        logging.error(f"County shapefile not found at {county_filepath}. Skipping county-level analysis.")
        return

    # --- 2. Data Preparation ---
    logging.info("Step 1/5: Preparing data for dual-category county analysis...")
    
    # Filter for the two distinct sets of hospitals
    hospitals_ai = hospital_gdf[hospital_gdf['ai_flag'] == 1].copy()
    hospitals_robo = hospital_gdf[hospital_gdf['robo_flag'] == 1].copy()
    
    logging.info(f"Found {len(hospitals_ai)} AI-enabled hospitals for analysis.")
    logging.info(f"Found {len(hospitals_robo)} Robotics-enabled hospitals for analysis.")

    # Prepare population data: add County FIPS and pre-calculate CBG centroids in radians
    pop_gdf['COUNTYFP'] = pop_gdf['GEOID'].str[:5]
    cbg_centroids_wgs84 = pop_gdf.to_crs(AEA_CRS).geometry.centroid.to_crs(WGS84_CRS)
    
    # Process coordinates separately and then combine
    y_coords = cbg_centroids_wgs84.apply(lambda p: p.y)
    x_coords = cbg_centroids_wgs84.apply(lambda p: p.x)
    
    # Convert to radians separately
    y_rad = np.deg2rad(y_coords)
    x_rad = np.deg2rad(x_coords)
    
    # Combine into a list of tuples
    pop_gdf['coords_rad'] = list(zip(y_rad, x_rad))
    
    # --- 3. Run Calculations for Each Category ---
    logging.info("Step 2/5: Calculating county-level metrics...")

    logging.info("...calculating for AI-enabled hospitals.")
    ai_times = _calculate_weighted_drive_time_by_county(
        pop_gdf, hospitals_ai, AVG_SPEED_MPH, EARTH_RADIUS_MILES
    )
    
    if ai_times is None or len(ai_times) == 0:
        logging.warning("No valid results for AI-enabled hospitals. Creating empty Series.")
        ai_times = pd.Series(dtype=float, name='avg_drive_time_AI_min')
    else:
        logging.info(f"Successfully calculated AI drive times for {len(ai_times)} counties.")

    logging.info("...calculating for Robotics-enabled hospitals.")
    robo_times = _calculate_weighted_drive_time_by_county(
        pop_gdf, hospitals_robo, AVG_SPEED_MPH, EARTH_RADIUS_MILES
    )
    
    if robo_times is None or len(robo_times) == 0:
        logging.warning("No valid results for Robotics-enabled hospitals. Creating empty Series.")
        robo_times = pd.Series(dtype=float, name='avg_drive_time_Robotics_min')
    else:
        logging.info(f"Successfully calculated Robotics drive times for {len(robo_times)} counties.")

    # --- 4. Merge Results ---
    logging.info("Step 3/5: Combining results into a final county DataFrame...")
    
    # Make sure we have data to work with
    if len(ai_times) == 0 and len(robo_times) == 0:
        logging.error("No drive time data calculated for either category. Cannot proceed.")
        # Create a minimal dataframe to avoid completely blank output
        error_df = pd.DataFrame({'error': ['No valid drive time data calculated']})
        error_df.to_csv(output_dir / "county_level_dual_access_analysis_ERROR.csv", index=False)
        return
    
    # FIX: Create results DataFrame properly to avoid ambiguous COUNTYFP
    if len(ai_times) > 0:
        # Reset index to make COUNTYFP a column
        results_df = pd.DataFrame({
            'avg_drive_time_AI_min': ai_times,
            'avg_drive_time_Robotics_min': robo_times if len(robo_times) > 0 else np.nan
        })
        results_df = results_df.reset_index().rename(columns={'index': 'COUNTYFP'})
    elif len(robo_times) > 0:
        results_df = pd.DataFrame({
            'avg_drive_time_AI_min': np.nan,
            'avg_drive_time_Robotics_min': robo_times
        })
        results_df = results_df.reset_index().rename(columns={'index': 'COUNTYFP'})
    else:
        # This is a fallback but shouldn't happen based on earlier checks
        results_df = pd.DataFrame(columns=['COUNTYFP', 'avg_drive_time_AI_min', 'avg_drive_time_Robotics_min'])
    
    # Ensure COUNTYFP is a string and properly formatted
    results_df['COUNTYFP'] = results_df['COUNTYFP'].astype(str).str.zfill(5)
    
    logging.info(f"Results DataFrame created with {len(results_df)} counties.")

    # Load county boundaries and merge everything together
    county_gdf = gpd.read_file(county_filepath)
    logging.info(f"Loaded county boundaries with {len(county_gdf)} counties.")
    
    # Ensure COUNTYFP in county_gdf is properly formatted for joining
    county_gdf['COUNTYFP'] = county_gdf['GEOID'].astype(str).str.zfill(5)
    
    # Calculate county population totals
    county_total_pop = pop_gdf.groupby('COUNTYFP')['POPULATION'].sum().reset_index()
    county_total_pop = county_total_pop.rename(columns={'POPULATION': 'total_population'})
    logging.info(f"Calculated population totals for {len(county_total_pop)} counties.")

    # Now merge should work without ambiguity
    logging.info("Merging county boundaries with drive time results...")
    final_county_df = pd.merge(county_gdf, results_df, on='COUNTYFP', how='outer')
    logging.info(f"After first merge: {len(final_county_df)} counties.")
    
    logging.info("Merging with population data...")
    final_county_df = pd.merge(final_county_df, county_total_pop, on='COUNTYFP', how='outer')
    logging.info(f"After second merge: {len(final_county_df)} counties.")
    
    # --- 5. Finalize and Save ---
    logging.info("Step 4/5: Cleaning and finalizing the dataset...")

    # Convert population to integer where possible, but keep NaN values
    final_county_df['total_population'] = pd.to_numeric(final_county_df['total_population'], errors='coerce')

    # Rename COUNTYFP to county_fips for consistency with other datasets
    if 'COUNTYFP' in final_county_df.columns:
        final_county_df.rename(columns={'COUNTYFP': 'county_fips'}, inplace=True)
        
    # Ensure county_fips is always a 5-digit string with leading zeros
    final_county_df['county_fips'] = final_county_df['county_fips'].astype(str).str.zfill(5)

    # Select and reorder columns, keeping all rows
    output_cols = [
        'STATE_NAME', 'NAME', 'county_fips',  # Changed from COUNTYFP to county_fips 
        'avg_drive_time_AI_min', 'avg_drive_time_Robotics_min', 'total_population'
    ]

    # Only keep columns that exist in the dataframe
    existing_cols = [col for col in output_cols if col in final_county_df.columns]
    final_county_df = final_county_df.reindex(columns=existing_cols + ['geometry'])

    # Rename columns
    rename_dict = {'STATE_NAME': 'state', 'NAME': 'county_name'}
    rename_dict = {k: v for k, v in rename_dict.items() if k in final_county_df.columns}
    final_county_df.rename(columns=rename_dict, inplace=True)
    
    logging.info(f"Final dataset has {len(final_county_df)} counties with {final_county_df.notna().sum().sum()} non-null values.")
    
    logging.info("Step 5/5: Displaying results and saving to CSV...")
    
    # Display top counties by access time (if we have that data)
    if 'avg_drive_time_AI_min' in final_county_df.columns and final_county_df['avg_drive_time_AI_min'].notna().any():
        print("\n\n" + "="*95)
        print("      Counties with Shortest Pop-Weighted Avg. Drive Time to ANY AI-ENABLED Hospital")
        print("="*95)
        
        # FIXED: Check which columns actually exist before trying to display them
        display_cols = []
        if 'state' in final_county_df.columns:
            display_cols.append('state')
        elif 'STATE_NAME' in final_county_df.columns:
            display_cols.append('STATE_NAME')
            
        if 'county_name' in final_county_df.columns:
            display_cols.append('county_name')
        elif 'NAME' in final_county_df.columns:
            display_cols.append('NAME')
            
        display_cols.append('avg_drive_time_AI_min')
        
        # Only try to print if we have valid columns
        if len(display_cols) > 1:  # We need at least county name or state plus the time
            print(final_county_df.sort_values('avg_drive_time_AI_min')[display_cols].head(10).to_string(index=False))
        else:
            print("(Cannot display detailed county information - column names not available)")
            print(final_county_df[['avg_drive_time_AI_min']].sort_values('avg_drive_time_AI_min').head(10))
    print("="*95)

    # Save to a single CSV - drop geometry column to avoid serialization issues
    csv_path = output_dir / "county_level_dual_access_analysis.csv"
    final_county_df_for_csv = final_county_df.drop(columns=['geometry'], errors='ignore')
    
    # Ensure we save even with partial data
    final_county_df_for_csv.to_csv(csv_path, index=False, float_format='%.2f')
    logging.info(f"Successfully saved detailed county-level access analysis for AI and Robotics to:\n{csv_path}")
    
    # Also save a simpler version with just the core results in case there are issues
    # Update the column name in the list as well
    core_cols = [col for col in ['county_fips', 'county_name', 'state', 'avg_drive_time_AI_min', 'avg_drive_time_Robotics_min'] 
                 if col in final_county_df_for_csv.columns]
    final_county_df_for_csv[core_cols].to_csv(output_dir / "county_level_core_results.csv", index=False, float_format='%.2f')
    logging.info(f"Also saved simplified core results to: {output_dir / 'county_level_core_results.csv'}")

# ======================= MAIN EXECUTION ==========================
def main():
    """Main function to orchestrate the analysis workflow."""
    setup_environment()
    engine = connect_to_db()

    hospital_df = load_hospital_data(engine)
    pop_gdf = load_population_data(POP_FILE)

    hospital_gdf_preprocessed = preprocess_data(hospital_df)
    hospital_gdf_cleaned = calculate_proximity_metrics(hospital_gdf_preprocessed)

    # Run the original state-level analysis
    analyze_state_level_access(hospital_gdf_cleaned, pop_gdf, STATES_FILE, BASE_DIR)
    
    # --- NEW: Run the detailed county-level analysis for both AI and Robotics ---
    analyze_county_level_access_detailed(hospital_gdf_cleaned, pop_gdf, COUNTY_FILE, BASE_DIR)
    
    # Run the poster map generation
    generate_poster_map(hospital_gdf_cleaned, pop_gdf)

    try:
        output_path = BASE_DIR / "hospital_ai_robotics_enriched.parquet"
        # Drop temporary columns before saving final output
        final_df = pd.DataFrame(hospital_gdf_cleaned.drop(columns=['geometry', 'county_fips_str', 'STATEFP'], errors='ignore'))
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