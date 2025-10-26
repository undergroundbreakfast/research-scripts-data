#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
# ==============================================================================
#  geospatial_hospital_ai_robotics_analysis.py
#  Author:  Aaron Johnson
#  Last updated: 2024-06-15
#  VERSION 4.0 - Revised based on initial run logs
#
#  End-to-end workflow analyzing spatial patterns of Generative AI and
#  robotics adoption in US hospitals (AHA 2024). This script performs:
#   1. Proximity analysis (nearest neighbor distances).
#   2. Population coverage analysis for AI/robotics hospitals.
#   3. Logistic regression to model adoption drivers.
#   4. Hot spot (LISA) analysis to find spatial clusters of adoption.
#   5. Generates publication-quality maps and figures.
# ==============================================================================
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

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

# ======================= CONFIGURATION ==========================
# --- Set Paths ---
BASE_DIR = Path(__file__).resolve().parent
SHAPE_DIR = BASE_DIR / "shapefiles"
FIG_DIR = BASE_DIR / "figures"
LOG_FILE = BASE_DIR / "analysis_log.txt"
CX_CACHE = BASE_DIR / "tile_cache"

# --- External Data Files (USER MUST PROVIDE) ---
# Download from https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
# The unzipped folder should be placed in the SHAPE_DIR
# COUNTY_FILE = SHAPE_DIR / "tl_2024_us_county" / "tl_2024_us_county.shp"
COUNTY_FILE = SHAPE_DIR / "tl_2024_us_county.shp"  # Custom file with 'GEOID' column
# Census Block Group file with a 'POPULATION' column, from a source like NHGIS.org
#POP_FILE = SHAPE_DIR / "us_bg2020_pop.geojson"
POP_FILE = SHAPE_DIR / "USA_BlockGroups_2020Pop.geojson"  # Custom file with 'POPULATION' column

# --- Analysis Parameters ---
DRIVE_MINUTES = 30
AVG_SPEED_MPH = 40.0
RADIUS_MILES = AVG_SPEED_MPH * DRIVE_MINUTES / 60.0

# --- Projections ---
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

def preprocess_data(df):
    """Cleans data, creates analysis columns, and returns a GeoDataFrame."""
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

    logging.info("Value counts for technology adoption:")
    logging.info(f"\n{df['ai_flag'].value_counts(dropna=False).rename('AI Adoption (any)')}")
    logging.info(f"\n{df['robo_flag'].value_counts(dropna=False).rename('Robotics Adoption')}")
    logging.info(f"\n{df['tech_type'].value_counts(dropna=False).rename('Technology Category')}")

    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=WGS84_CRS
    )
    logging.info("Created GeoDataFrame with hospital locations.")
    return gdf

# ======================= ANALYSIS FUNCTIONS ==========================
def calculate_proximity_metrics(gdf):
    """Calculates distance to nearest hospital and average to 3 nearest."""
    logging.info("\n" + "="*20 + " PROXIMITY ANALYSIS " + "="*20)
    if len(gdf) < 4:
        logging.warning("Not enough hospitals (< 4) to calculate k=3 neighbors. Skipping.")
        gdf['nearest_miles'], gdf['k3_avg_miles'] = np.nan, np.nan
        return gdf

    coords_rad = np.deg2rad(gdf[['latitude', 'longitude']].values)
    tree = BallTree(coords_rad, metric='haversine')
    distances, _ = tree.query(coords_rad, k=4)
    earth_radius_miles = 3958.8
    distances_miles = distances * earth_radius_miles

    gdf['nearest_miles'] = distances_miles[:, 1]
    gdf['k3_avg_miles'] = distances_miles[:, 1:4].mean(axis=1)

    logging.info("Calculated proximity metrics (nearest hospital, k3 average).")
    logging.info(f"Mean distance to nearest hospital: {gdf['nearest_miles'].mean():.2f} miles (SD: {gdf['nearest_miles'].std():.2f})")
    logging.info(f"Mean avg distance to 3 nearest: {gdf['k3_avg_miles'].mean():.2f} miles (SD: {gdf['k3_avg_miles'].std():.2f})")
    return gdf

def analyze_population_coverage(hospital_gdf):
    """Analyzes population within a 30-min drive of AI/Robotics hospitals."""
    logging.info("\n" + "="*20 + " POPULATION COVERAGE ANALYSIS " + "="*20)
    if not POP_FILE.exists():
        logging.warning(f"Population file not found at {POP_FILE}. Skipping coverage analysis.")
        return
    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}, skipping coverage map.")
        return

    try:
        pop_gdf = gpd.read_file(POP_FILE)
        
        # Find population column regardless of case
        pop_col = None
        for col in pop_gdf.columns:
            if col.lower() == 'population':
                pop_col = col
                break
        
        if not pop_col:
            logging.error(f"No population column found in {POP_FILE}. Available columns: {list(pop_gdf.columns)}")
            return
            
        logging.info(f"Using '{pop_col}' as the population column")
        pop_gdf['POPULATION'] = pd.to_numeric(pop_gdf[pop_col], errors='coerce').fillna(0)
        
        hosp_proj = hospital_gdf.to_crs(AEA_CRS)
        pop_proj = pop_gdf.to_crs(AEA_CRS)
        total_population = pop_proj['POPULATION'].sum()
        logging.info(f"Loaded {len(pop_gdf)} population block groups. Total US population: {total_population:,.0f}")
        logging.info(f"Using {DRIVE_MINUTES}-minute drive time proxy ({RADIUS_MILES:.1f} mile buffer).")

        def calculate_coverage(tech_gdf, tech_name):
            if tech_gdf.empty:
                logging.warning(f"No hospitals found for '{tech_name}'. Coverage is 0.")
                return None, 0, 0.0
            buffer_radius_meters = RADIUS_MILES * 1609.34
            coverage_area = tech_gdf.buffer(buffer_radius_meters).unary_union
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
            try:
                cx.set_cache_dir(str(CX_CACHE))
                cx.add_basemap(ax, crs=AEA_CRS, source=cx.providers.Stamen.TonerLite, zoom='auto')
            except Exception as e1:
                logging.warning(f"Primary basemap source failed: {e1}")
                try:
                    logging.info("Trying alternative basemap source (OpenStreetMap)...")
                    cx.add_basemap(ax, crs=AEA_CRS, source=cx.providers.OpenStreetMap.Mapnik, zoom='auto')
                except Exception as e2:
                    logging.warning(f"Alternative basemap source failed: {e2}")
        fig_path = FIG_DIR / "population_coverage_map.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved population coverage map to {fig_path}")
        plt.close(fig)

    except Exception as e:
        logging.error(f"Error during population coverage analysis: {e}", exc_info=True)

def model_adoption_drivers(gdf):
    """Models AI/Robotics adoption based on competition and hospital size."""
    logging.info("\n" + "="*20 + " MODELING ADOPTION DRIVERS (REVISED) " + "="*20)
    
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
        logging.info("NOTE: A negative coefficient for 'k3_avg_miles' suggests that shorter distances (more competition) are associated with higher odds of adoption.")
        logging.info("Classification Report:\n" + classification_report(y, model.predict(X_scaled), digits=3, zero_division=0))

    # REVISED: Use a more stable distance metric and add a control variable (bed size)
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
        county_gdf = gpd.read_file(COUNTY_FILE).to_crs(AEA_CRS)
        if 'GEOID' not in county_gdf.columns:
            logging.error("County shapefile must have a 'GEOID' column.")
            return

        hosp_proj = hospital_gdf.to_crs(AEA_CRS)
        hosp_with_county = gpd.sjoin(hosp_proj, county_gdf[['GEOID', 'geometry']], how='left', predicate='within')
        county_agg = hosp_with_county.groupby('GEOID').agg(
            ai_count=('ai_flag', 'sum'),
            robo_count=('robo_flag', 'sum'),
            ai_intensity_sum=('ai_intensity', 'sum'),
            hospital_count=('hospital_id', 'count')
        ).reset_index()
        county_agg['ai_rate'] = county_agg['ai_count'] / county_agg['hospital_count']
        county_agg['robo_rate'] = county_agg['robo_count'] / county_agg['hospital_count']
        analysis_gdf = county_gdf.merge(county_agg, on='GEOID', how='left').fillna(0)

        def run_lisa_and_plot(gdf, column, title):
            logging.info(f"--- Running LISA for: {column} ---")
            y = gdf[column]
            w = libpysal.weights.Queen.from_dataframe(gdf, use_index=True, silence_warnings=True)
            w.transform = 'r'
            
            try:
                lisa = esda.Moran_Local(y, w)
                fig, ax = plt.subplots(figsize=(15, 10))
                lisa_cluster(lisa, gdf, p=0.05, ax=ax, legend_kwds={'loc': 'lower left'})
                ax.set_title(title, fontsize=16)
                ax.set_axis_off()
                
                # Try to add basemap with error handling and fallback sources
                if CONTEXTILY_AVAILABLE:
                    try:
                        # Configure cache
                        cx.set_cache_dir(str(CX_CACHE))
                        # Try primary source
                        cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.Stamen.TonerLite)
                    except Exception as e1:
                        logging.warning(f"Primary basemap source failed: {e1}")
                        try:
                            # Try alternative source 1 - OpenStreetMap
                            logging.info("Trying alternative basemap source (OpenStreetMap)...")
                            cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.OpenStreetMap.Mapnik)
                        except Exception as e2:
                            logging.warning(f"Alternative basemap source 1 failed: {e2}")
                            try:
                                # Try alternative source 2 - CartoDB
                                logging.info("Trying alternative basemap source (CartoDB)...")
                                cx.add_basemap(ax, crs=gdf.crs, source=cx.providers.CartoDB.Positron)
                            except Exception as e3:
                                logging.warning(f"All basemap sources failed. Creating map without basemap.")
                
                fig_path = FIG_DIR / f"lisa_hotspot_{column}.png"
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved LISA cluster map to {fig_path}")
                plt.close(fig)
            except Exception as e:
                logging.error(f"Error during LISA analysis for {column}: {e}")
                plt.close('all')  # Ensure any open figure is closed

        run_lisa_and_plot(analysis_gdf, 'ai_rate', 'Hot Spots of AI Adoption Rate by County')
        run_lisa_and_plot(analysis_gdf, 'robo_rate', 'Hot Spots of Robotics Adoption Rate by County')
        run_lisa_and_plot(analysis_gdf, 'ai_intensity_sum', 'Hot Spots of AI Adoption Intensity by County')
    except Exception as e:
        logging.error(f"Error during hot spot analysis: {e}", exc_info=True)


def generate_summary_visualizations(gdf):
    """Creates general-purpose visualizations for the final report."""
    logging.info("\n" + "="*20 + " GENERATING SUMMARY VISUALIZATIONS " + "="*20)
    sns.set_theme(style="whitegrid", palette="viridis")

    # 1. Map of all hospitals by technology type (FIXED)
    if not COUNTY_FILE.exists():
        logging.warning(f"County file not found at {COUNTY_FILE}, skipping distribution map.")
    else:
        try:
            us_counties = gpd.read_file(COUNTY_FILE)
            us_main = us_counties[~us_counties['STATEFP'].isin(['02', '15', '60', '66', '69', '72', '78'])]
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            us_main.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5)
            gdf.plot(
                ax=ax, column='tech_type', categorical=True, legend=True, markersize=15,
                alpha=0.7, legend_kwds={'title': "Technology Type", 'loc': 'lower left'}
            )
            ax.set_title('Distribution of US Hospitals by AI and Robotics Adoption', fontsize=16)
            ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
            ax.set_axis_off()
            fig_path = FIG_DIR / "hospital_distribution_by_tech.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved hospital distribution map to {fig_path}")
            plt.close(fig)
        except Exception as e:
            logging.error(f"Could not generate distribution map: {e}")

    # 2. Histogram of nearest hospital distances
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=gdf, x='nearest_miles', bins=50, kde=True, ax=ax)
    ax.set_title('Distribution of Distances to Nearest Hospital')
    ax.set_xlabel('Distance to Nearest Hospital (miles)'); ax.set_ylabel('Number of Hospitals')
    ax.axvline(gdf['nearest_miles'].median(), color='red', linestyle='--', label=f"Median: {gdf['nearest_miles'].median():.1f} mi")
    ax.legend()
    fig_path = FIG_DIR / "nearest_distance_histogram.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved distance histogram to {fig_path}")
    plt.close(fig)

    # 3. Boxplot of distance vs. adoption
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    sns.boxplot(data=gdf, x=gdf['ai_flag'].astype('category'), y='nearest_miles', ax=axes[0])
    axes[0].set_title('AI Adoption vs. Proximity'); axes[0].set_xlabel('AI Adopted (1=Yes, 0=No)'); axes[0].set_ylabel('Distance to Nearest Hospital (miles)')
    sns.boxplot(data=gdf, x=gdf['robo_flag'].astype('category'), y='nearest_miles', ax=axes[1])
    axes[1].set_title('Robotics Adoption vs. Proximity'); axes[1].set_xlabel('Robotics Adopted (1=Yes, 0=No)'); axes[1].set_ylabel('')
    plt.suptitle('Competition and Technology Adoption', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_path = FIG_DIR / "adoption_vs_distance_boxplot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved adoption vs. distance boxplot to {fig_path}")
    plt.close(fig)


# ======================= MAIN EXECUTION ==========================
def main():
    """Main function to orchestrate the analysis workflow."""
    setup_environment()
    engine = connect_to_db()
    hospital_df = load_hospital_data(engine)
    hospital_gdf = preprocess_data(hospital_df)
    hospital_gdf = calculate_proximity_metrics(hospital_gdf)
    
    # Add this code after you've called calculate_proximity_metrics()
    # and have the nearest_miles column available

    # Print the hospitals with the largest distances to help identify the outliers
    print("\nExamining potential outliers:")
    extreme_outliers = hospital_gdf.sort_values('nearest_miles', ascending=False).head(5)
    if 'mname' in extreme_outliers.columns:
        print(extreme_outliers[['mname', 'city', 'state', 'nearest_miles', 'k3_avg_miles']].to_string())
    else:
        print(extreme_outliers[['nearest_miles', 'k3_avg_miles']].to_string())

    # Create a targeted filter for just the extreme outliers
    # Using 1500 miles as a reasonable upper limit - this is much higher than typical distances
    # even for remote locations in Alaska/Hawaii, but will catch the 2000 and 8000 mile errors
    extreme_outlier_threshold = 1500  # miles

    # Count and identify the extreme outliers
    extreme_count = (hospital_gdf['nearest_miles'] > extreme_outlier_threshold).sum()
    if extreme_count > 0:
        extreme_hospitals = hospital_gdf[hospital_gdf['nearest_miles'] > extreme_outlier_threshold]
        print(f"\nRemoving {extreme_count} extreme outliers with nearest_miles > {extreme_outlier_threshold} miles:")
        if 'mname' in extreme_hospitals.columns:
            for idx, row in extreme_hospitals.iterrows():
                print(f"Removing: {row.get('mname', 'Unknown')} in {row.get('city', 'Unknown')}, {row.get('state', 'Unknown')} - {row['nearest_miles']:.2f} miles")
        
        # Filter out just these extreme outliers
        hospital_gdf = hospital_gdf[hospital_gdf['nearest_miles'] <= extreme_outlier_threshold].copy()
        print(f"Dataset now contains {len(hospital_gdf)} hospitals after removing extreme outliers.")

        # Report the new maximum distance
        max_dist = hospital_gdf['nearest_miles'].max()
        print(f"New maximum nearest-hospital distance: {max_dist:.2f} miles")
    else:
        print("No extreme outliers found with the current threshold.")

    analyze_population_coverage(hospital_gdf)
    model_adoption_drivers(hospital_gdf)
    perform_hotspot_analysis(hospital_gdf)
    generate_summary_visualizations(hospital_gdf)

    try:
        output_path = BASE_DIR / "hospital_ai_robotics_enriched.parquet"
        final_df = pd.DataFrame(hospital_gdf.drop(columns='geometry'))
        final_df['geometry_wkt'] = hospital_gdf.geometry.to_wkt()
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