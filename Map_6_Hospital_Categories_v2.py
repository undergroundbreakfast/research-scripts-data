#!/usr/bin/env python3

"""
Revised May 18th, 2025

This script loads a U.S. county shapefile, retrieves hospital category data from a PostgreSQL database view,
and then displays three separate windows:
  1. Contiguous US counties.
  2. Alaska (reprojected to Alaska Albers so that all areas appear together).
  3. Hawaii.

In each window, counties are colored based on their hospital category (1-6), with 1 being the least intense (no hospital)
and 6 being the most intense (multiple hospitals with BOTH AI and robotics).
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------
# 1. Load the shapefile and separate regions
# -------------------------------
shapefile_path = "/Users/aaronjohnson/Documents/python_runtime/tl_2024_us_county.shp"
gdf = gpd.read_file(shapefile_path)

# Filter regions by STATEFP codes:
#   - Contiguous US: Exclude Alaska ("02"), Hawaii ("15"), and territories ("60", "66", "69", "72", "78")
#   - Alaska: STATEFP == "02"
#   - Hawaii: STATEFP == "15"
contiguous_gdf = gdf[~gdf["STATEFP"].isin(["02", "15", "60", "66", "69", "72", "78"])]
alaska_gdf = gdf[gdf["STATEFP"] == "02"]
hawaii_gdf = gdf[gdf["STATEFP"] == "15"]

# -------------------------------
# 2. Connect to PostgreSQL and retrieve hospital category data
# -------------------------------
host = 'localhost'
database = 'Research_TEST'
user = 'postgres'
password = os.getenv("POSTGRESQL_KEY")  # Make sure this is set in your environment

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}/{database}")

query = """
SELECT
    county_fips AS county_fips_code,
    county_category
FROM vw_conceptual_model;
"""

hospital_data = pd.read_sql(query, engine)

# Ensure the FIPS codes are five characters long (with leading zeros)
hospital_data["county_fips_code"] = hospital_data["county_fips_code"].astype(str).str.zfill(5)
# Convert the county category to numeric and replace missing values with 1 (assuming missing means "No Hospital")
hospital_data["county_category"] = pd.to_numeric(hospital_data["county_category"], errors="coerce").fillna(1)

# -------------------------------
# 3. Merge hospital data with each region's GeoDataFrame
# -------------------------------
contiguous_merged = contiguous_gdf.merge(hospital_data, left_on="GEOID", right_on="county_fips_code", how="left")
alaska_merged = alaska_gdf.merge(hospital_data, left_on="GEOID", right_on="county_fips_code", how="left")
hawaii_merged = hawaii_gdf.merge(hospital_data, left_on="GEOID", right_on="county_fips_code", how="left")

# -------------------------------
# 4. Plot the contiguous US in its own window
# -------------------------------
# Use a larger figure with better proportions for a 16" MacBook Pro
fig1, ax1 = plt.subplots(figsize=(14, 10))  # Increased height for legend room

# Create a custom colormap with white-gray-blue transition

# Define custom colors for the 6 categories
colors = [
    'white',          # 1. No Hospital (white)
    '#D3D3D3',        # 2. Hospital (No Robotics/AI) (light gray)
    '#A9A9A9',        # 3. Hospital w/ Robotics (medium gray)
    '#9ecae1',        # 4. Hospital w/ AI (light blue)
    '#4292c6',        # 5. Hospital w/ BOTH (medium blue)
    '#084594'         # 6. Multiple w/ BOTH (dark blue)
]

# Create custom colormap
custom_cmap = LinearSegmentedColormap.from_list("white_gray_blue", colors, N=6)
norm = plt.Normalize(vmin=1, vmax=6)

# Plot the map with new colormap
contiguous_merged.plot(column="county_category", 
                      ax=ax1, 
                      cmap=custom_cmap,
                      norm=norm,
                      edgecolor="black", 
                      linewidth=0.2,
                      legend=False)

# Add title with better positioning
ax1.set_title("County Hospital Categories – Contiguous US", 
             fontsize=16, 
             pad=20)

# Remove x and y labels completely - they're not needed for a geographic map
ax1.set_xlabel("")
ax1.set_ylabel("")

# Set extent to focus on contiguous US with improved framing
ax1.set_xlim([-125, -66.5])
ax1.set_ylim([23.5, 49.5])

# Remove axis ticks for a cleaner look
ax1.set_xticks([])
ax1.set_yticks([])

# Add legend with detailed category descriptions
legend_elements = [
    plt.Rectangle((0,0), 1, 1, fc=colors[0], edgecolor='lightgray', label='1. No Hospital'),
    plt.Rectangle((0,0), 1, 1, fc=colors[1], edgecolor='none', label='2. Hospital (No Robotics/AI)'),
    plt.Rectangle((0,0), 1, 1, fc=colors[2], edgecolor='none', label='3. Hospital w/ Robotics'),
    plt.Rectangle((0,0), 1, 1, fc=colors[3], edgecolor='none', label='4. Hospital w/ AI'),
    plt.Rectangle((0,0), 1, 1, fc=colors[4], edgecolor='none', label='5. Hospital w/ BOTH'),
    plt.Rectangle((0,0), 1, 1, fc=colors[5], edgecolor='none', label='6. Multiple w/ BOTH')
]

# First apply tight layout to position the map
fig1.tight_layout(rect=[0.02, 0.12, 0.98, 0.95])  # Increased bottom margin

# Add legend below the map with more spacing from bottom
legend = ax1.legend(handles=legend_elements, 
                   loc='upper center', 
                   bbox_to_anchor=(0.5, -0.02),  # Added more spacing from bottom (was -0.05)
                   ncol=2,  # Two columns for more compact layout
                   fontsize=10, 
                   frameon=True,  
                   facecolor='white',
                   edgecolor='lightgray',
                   title="Hospital Category")

# Set figure background to white
fig1.patch.set_facecolor('white')

# Save the figure with high resolution for publication
fig1.savefig("county_hospital_categories_map.png", dpi=300, bbox_inches='tight')

# -------------------------------
# 5. Plot Alaska in its own window
#    Reproject to an Alaska-appropriate projection to avoid antimeridian issues.
# -------------------------------
# EPSG:3338 is NAD83 / Alaska Albers, which is suitable for Alaska.
alaska_merged = alaska_merged.to_crs(epsg=3338)
fig2, ax2 = plt.subplots(figsize=(12, 8))
alaska_merged.plot(column="county_category", 
                  ax=ax2, 
                  cmap=custom_cmap,
                  norm=norm,
                  edgecolor="black", 
                  linewidth=0.2,
                  legend=False)
ax2.set_title("County Hospital Categories – Alaska")
ax2.set_xlabel("Easting (m)")
ax2.set_ylabel("Northing (m)")
# Set axis limits based on Alaska's total bounds with a small margin
minx, miny, maxx, maxy = alaska_merged.total_bounds
margin_x = (maxx - minx) * 0.05
margin_y = (maxy - miny) * 0.05
ax2.set_xlim(minx - margin_x, maxx + margin_x)
ax2.set_ylim(miny - margin_y, maxy + margin_y)

# -------------------------------
# 6. Plot Hawaii in its own window
# -------------------------------
fig3, ax3 = plt.subplots(figsize=(12, 8))
hawaii_merged.plot(column="county_category", 
                  ax=ax3, 
                  cmap=custom_cmap,
                  norm=norm,
                  edgecolor="black", 
                  linewidth=0.2,
                  legend=False)
ax3.set_title("County Hospital Categories – Hawaii")
ax3.set_xlabel("Longitude")
ax3.set_ylabel("Latitude")
# Use total bounds of the Hawaii data (with an added margin)
minx, miny, maxx, maxy = hawaii_merged.total_bounds
margin_x = (maxx - minx) * 0.1
margin_y = (maxy - miny) * 0.1
ax3.set_xlim(minx - margin_x, maxx + margin_x)
ax3.set_ylim(miny - margin_y, maxy + margin_y)

# -------------------------------
# 7. Display all windows
# -------------------------------
plt.show()