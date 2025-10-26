#!/usr/bin/env python3
# ==============================================================================
# Copyright (c) 2025 Aaron Johnson, Drexel University
# Licensed under the MIT License - see LICENSE file for details
# ==============================================================================
# ==============================================================================
#  generate_county_poster.py
#  Author:  Aaron Johnson
#  Last updated: 2025-07-19
#
#  VERSION: Merged
#  This script combines two previous workflows. It fetches county-level hospital
#  technology adoption data from a PostgreSQL database and generates a
#  36" x 48" high-resolution poster map of the contiguous United States.
#  The map visualizes six distinct categories of technology adoption using a
#  custom color scheme and is styled for conference presentation.
# ==============================================================================

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# --- Core Libraries ---
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text  # Add text import here

# --- Visualization Libraries ---
import matplotlib
matplotlib.use('Cairo') # Use Cairo backend for high-quality, non-interactive output
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Geospatial & Poster-Specific Libraries ---
try:
    from matplotlib_scalebar.scalebar import ScaleBar
    SCALEBAR_AVAILABLE = True
except ImportError:
    SCALEBAR_AVAILABLE = False
    print("Warning: 'matplotlib-scalebar' not found. Scale bar will not be added to poster. Run: pip install matplotlib-scalebar")

# ======================= CONFIGURATION ==========================
# --- Directory Setup ---
BASE_DIR = Path(__file__).resolve().parent
SHAPE_DIR = BASE_DIR / "shapefiles"
FIG_DIR = BASE_DIR / "figures"
LOG_FILE = BASE_DIR / "poster_generation_log.txt"

# --- File Paths ---
# Assumes shapefiles are in a 'shapefiles' subdirectory
COUNTY_SHAPEFILE = SHAPE_DIR / "tl_2024_us_county.shp"
STATE_SHAPEFILE = SHAPE_DIR / "tl_2023_us_state.shp" # For clean state outlines on the poster

# --- Map & Poster Specifications ---
# Use Albers Equal Area projection designed for US maps
AEA_CRS = "EPSG:5070"
POSTER_W_IN = 48.0
POSTER_H_IN = 36.0
POSTER_DPI = 150 # 150 DPI is a good balance of quality and file size for a large poster

# ======================= SETUP ================================
def setup_environment():
    """Create necessary directories and configure logging."""
    for p in (SHAPE_DIR, FIG_DIR):
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
    logging.info(f"POSTER GENERATION RUN STARTED: {datetime.now().isoformat()}")
    logging.info("=" * 60)

# ======================= DATA LOADING =======================
def connect_to_db():
    """Establishes a connection to the PostgreSQL database."""
    try:
        db_password = os.getenv("POSTGRESQL_KEY")
        if not db_password:
            raise ValueError("FATAL: POSTGRESQL_KEY environment variable not set.")
        engine = create_engine(
            f"postgresql+psycopg2://{os.getenv('PGUSER', 'postgres')}:{db_password}@{os.getenv('PGHOST', 'localhost')}/{os.getenv('PGDATABASE', 'Research_TEST')}",
            pool_pre_ping=True, connect_args={"connect_timeout": 10},
        )
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # Use imported text function here
        logging.info(f"Postgres connection successful: {engine.url.host}/{engine.url.database}")
        return engine
    except Exception as e:
        logging.exception("FATAL: Failed to connect to PostgreSQL database.")
        sys.exit(1)

def load_hospital_category_data(engine):
    """Loads hospital category data from the PostgreSQL view."""
    logging.info("Querying PostgreSQL for county hospital categories...")
    query = """
    SELECT
        county_fips AS county_fips_code,
        county_category
    FROM vw_conceptual_model;
    """
    try:
        hospital_data = pd.read_sql(query, engine)
        # Ensure FIPS codes are five characters long (with leading zeros)
        hospital_data["county_fips_code"] = hospital_data["county_fips_code"].astype(str).str.zfill(5)
        # Convert category to numeric and fill missing values with 1 (No Hospital)
        hospital_data["county_category"] = pd.to_numeric(hospital_data["county_category"], errors="coerce").fillna(1)
        logging.info(f"Successfully loaded and processed {len(hospital_data)} county category records.")
        return hospital_data
    except Exception as e:
        logging.exception("FATAL: Error executing query for hospital category data.")
        sys.exit(1)

def load_geospatial_data():
    """Loads the county and state shapefiles."""
    logging.info("Loading geospatial data (counties and states)...")
    if not COUNTY_SHAPEFILE.exists():
        logging.error(f"FATAL: County shapefile not found at: {COUNTY_SHAPEFILE}")
        sys.exit(1)
    if not STATE_SHAPEFILE.exists():
        logging.error(f"FATAL: State shapefile not found at: {STATE_SHAPEFILE}")
        sys.exit(1)

    county_gdf = gpd.read_file(COUNTY_SHAPEFILE)
    state_gdf = gpd.read_file(STATE_SHAPEFILE)
    logging.info(f"Loaded {len(county_gdf)} county geometries and {len(state_gdf)} state geometries.")
    return county_gdf, state_gdf

# ======================= POSTER GENERATION ==========================
def generate_county_category_poster(merged_gdf, states_gdf):
    """
    Generates a 36" x 48" poster map showing U.S. counties colored by
    hospital technology adoption category.
    """
    logging.info("\n" + "="*20 + " GENERATING 36x48 COUNTY CATEGORY POSTER MAP " + "="*20)

    # --- 1. Prepare Data for Plotting ---
    logging.info("Step 1/5: Preparing data for the poster map...")
    # Filter to contiguous US using STATEFP codes
    excluded_fips = ["02", "15", "60", "66", "69", "72", "78"] # AK, HI, and territories
    contiguous_counties = merged_gdf[~merged_gdf["STATEFP"].isin(excluded_fips)].copy()
    contiguous_states = states_gdf[~states_gdf["STUSPS"].isin(['AK', 'HI', 'PR', 'GU', 'AS', 'VI', 'MP'])].copy()

    # Reproject all data to the target Albers Equal Area CRS
    counties_proj = contiguous_counties.to_crs(AEA_CRS)
    states_proj = contiguous_states.to_crs(AEA_CRS)
    logging.info(f"Filtered to {len(counties_proj)} contiguous US counties and reprojected to {AEA_CRS}.")

    # --- 2. Define Colors and Normalization ---
    logging.info("Step 2/5: Defining custom color scheme and normalization...")
    colors = [
        '#FFFFFF',  # 1. No Hospital (white)
        '#D3D3D3',  # 2. Hospital (No Robotics/AI) (light gray)
        '#A9A9A9',  # 3. Hospital w/ Robotics (medium gray)
        '#9ecae1',  # 4. Hospital w/ AI (light blue)
        '#4292c6',  # 5. Hospital w/ BOTH (medium blue)
        '#084594'   # 6. Multiple w/ BOTH (dark blue)
    ]
    custom_cmap = LinearSegmentedColormap.from_list("white_gray_blue", colors, N=6)
    norm = plt.Normalize(vmin=1, vmax=6)

    # --- 3. Setup Poster Canvas and Plot Data ---
    logging.info("Step 3/5: Creating poster canvas and rendering map layers...")
    fig, ax = plt.subplots(1, 1, figsize=(POSTER_W_IN, POSTER_H_IN))
    ax.set_aspect('equal')

    # Plot the county data
    counties_proj.plot(
        ax=ax,
        column="county_category",
        cmap=custom_cmap,
        norm=norm,
        edgecolor="black",
        linewidth=0.15, # Thinner lines for a cleaner look on a dense map
        legend=False,
        rasterized=True # Improves performance and reduces file size for PDFs
    )

    # Overlay clean state borders for context
    states_proj.plot(ax=ax, facecolor='none', edgecolor='white', linewidth=0.8, alpha=0.9)

    # --- 4. Add Map Furniture (Titles, Legend, Scale Bar, Credits) ---
    logging.info("Step 4/5: Adding titles, legend, and other map elements...")
    ax.set_axis_off()

    # Add Title and Subtitle
    ax.set_title(
        "U.S. County-Level Hospital Technology Adoption",
        fontsize=48, pad=30, fontweight='bold'
    )
    ax.text(0.5, 0.96,
            "Analysis of AI and Robotics presence in U.S. Hospitals by County",
            ha='center', va='center', transform=ax.transAxes, fontsize=28)

    # Add Custom Legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, fc=colors[0], ec='darkgray', lw=0.5, label='1. No Hospital'),
        plt.Rectangle((0,0), 1, 1, fc=colors[1], label='2. Hospital (No Robotics/AI)'),
        plt.Rectangle((0,0), 1, 1, fc=colors[2], label='3. Hospital w/ Robotics Only'),
        plt.Rectangle((0,0), 1, 1, fc=colors[3], label='4. Hospital w/ AI Only'),
        plt.Rectangle((0,0), 1, 1, fc=colors[4], label='5. Hospital w/ Both AI & Robotics'),
        plt.Rectangle((0,0), 1, 1, fc=colors[5], label='6. Multiple Hospitals w/ Both')
    ]
    ax.legend(handles=legend_elements,
              loc='lower right',
              bbox_to_anchor=(0.99, 0.08), # Positioned to avoid credit text
              ncol=1,
              fontsize=20,
              title_fontsize=24,
              frameon=True,
              facecolor='white',
              edgecolor='black',
              title="Hospital Technology Category")

    # Add Scale Bar
    if SCALEBAR_AVAILABLE:
        try:
            # Map projection is in meters. We create a scale bar in km that represents ~200 miles.
            ax.add_artist(ScaleBar(
                dx=1, units='m', fixed_value=320, fixed_units='km',
                location='lower left', length_fraction=0.25, box_alpha=0.8,
                font_properties={'size': 18}, scale_loc='bottom', label="~200 miles"
            ))
            logging.info("Successfully added scale bar (320 km ≈ 200 miles).")
        except Exception as e:
            logging.warning(f"Scale bar creation failed: {e}. Skipping.")

    # Add Credit Text
    ax.text(0.99, 0.01,
            f"Author: Aaron Johnson | Data: AHA (2024), US Census (2024), vw_conceptual_model\nMap Projection: USA Contiguous Albers Equal Area Conic (EPSG:{AEA_CRS})\nConnecticut data is intentionally excluded from this map due to county-equivalent changes.",
            ha='right', va='bottom', transform=ax.transAxes, fontsize=14, color='gray')

    # --- 5. Save Final Poster ---
    logging.info("Step 5/5: Saving the final poster to PDF...")
    output_filename = FIG_DIR / "poster_map_county_hospital_categories.pdf"
    plt.savefig(
        output_filename,
        dpi=POSTER_DPI,
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.5
    )
    logging.info(f"\nSUCCESS: Poster map saved to:\n{output_filename}")
    plt.close(fig)


# ======================= MAIN EXECUTION ==========================
def main():
    """Main function to orchestrate the poster generation workflow."""
    setup_environment()
    engine = connect_to_db()

    # --- Load Data ---
    hospital_data_df = load_hospital_category_data(engine)
    county_gdf, state_gdf = load_geospatial_data()

    # --- Merge Geospatial and Attribute Data ---
    logging.info("Merging hospital category data with county geometries...")
    # The left join ensures all counties are kept for a complete map.
    merged_gdf = county_gdf.merge(
        hospital_data_df,
        left_on="GEOID",
        right_on="county_fips_code",
        how="left"
    )
    # After the merge, fill any counties that didn't have a match with category 1.
    # This is safer than relying on the .fillna in the data loading step.
    merged_gdf["county_category"] = merged_gdf["county_category"].fillna(1)
    logging.info("Merge complete. All counties now have a technology category.")

    # --- Generate the Poster ---
    generate_county_category_poster(merged_gdf, state_gdf)

    logging.info("=" * 60)
    logging.info("RUN COMPLETED")
    logging.info("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An unhandled exception occurred during the main execution.")
        sys.exit(1)