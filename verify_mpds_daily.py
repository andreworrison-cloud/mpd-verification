#!/usr/bin/env python3
"""
WPC MPD Verification Script (GitHub Actions Version with Archive)
Generates verification maps for yesterday's Mesoscale Precipitation Discussions
and builds a historical HTML archive. Images are clickable for full resolution.
"""

import os
import requests
import datetime
from datetime import datetime as dt
from collections import defaultdict
from io import BytesIO

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. Data Ingestion Functions ---

def fetch_iem_mpds(target_date):
    """Pulls WPC Mesoscale Precipitation Discussions from the IEM archive."""
    print(f"Fetching MPDs for {target_date.strftime('%Y-%m-%d')}...")
    yr = target_date.strftime('%Y')
    mo = target_date.strftime('%m')
    dy = target_date.strftime('%d')
    
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/gis/wpc_mpd.py?"
        f"year1={yr}&month1={mo}&day1={dy}&hour1=0&minute1=0&"
        f"year2={yr}&month2={mo}&day2={dy}&hour2=23&minute2=59"
    )
    response = requests.get(url)
    if response.status_code == 200:
        try:
            gdf = gpd.read_file(BytesIO(response.content))
            gdf['start_time'] = pd.to_datetime(gdf['ISSUE'])
            gdf['end_time'] = pd.to_datetime(gdf['EXPIRE'])
            return gdf
        except Exception as e:
            print(f"No MPDs found or error parsing: {e}")
            return gpd.GeoDataFrame()
    else:
        print("Failed to connect to IEM MPD API.")
        return gpd.GeoDataFrame()

def fetch_iem_ffws(start_date, end_date):
    """Pulls strictly Storm-Based FFW polygons from the IEM archive."""
    print(f"Fetching Storm-Based FFWs from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    yr1, mo1, dy1 = start_date.strftime('%Y'), start_date.strftime('%m'), start_date.strftime('%d')
    yr2, mo2, dy2 = end_date.strftime('%Y'), end_date.strftime('%m'), end_date.strftime('%d')
    
    url = (
        f"https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py?"
        f"year1={yr1}&month1={mo1}&day1={dy1}&hour1=0&minute1=0&"
        f"year2={yr2}&month2={mo2}&day2={dy2}&hour2=23&minute2=59&"
        f"limit0=yes&limit1=yes" 
    )
    response = requests.get(url)
    if response.status_code == 200:
        try:
            gdf = gpd.read_file(BytesIO(response.content))
            ffw_gdf = gdf[(gdf['PHENOM'] == 'FF') & (gdf['SIG'] == 'W')].copy()
            
            # Parse traditional issue and expiration times
            ffw_gdf['issue_time'] = pd.to_datetime(ffw_gdf['ISSUED'])
            ffw_gdf['expire_time'] = pd.to_datetime(ffw_gdf['EXPIRED'])
            
            # CRITICAL EXT FIX: Grab the actual product timestamp to catch Extensions 
            if 'UPDATED' in ffw_gdf.columns:
                ffw_gdf['product_time'] = pd.to_datetime(ffw_gdf['UPDATED'])
            else:
                ffw_gdf['product_time'] = ffw_gdf['issue_time'] # Fallback
                
            return ffw_gdf
        except Exception as e:
            print(f"Error parsing FFW shapefile: {e}")
            return gpd.GeoDataFrame()
    else:
        print("Failed to connect to IEM FFW API.")
        return gpd.GeoDataFrame()

# --- 2. Logic & Plotting Functions ---

def classify_ffw_polygons(mpd_gdf, ffw_gdf):
    """Classifies FFW polygons based on product issuance time and spatial overlap with an MPD."""
    ffw_gdf['plot_color'] = 'drop' # Default state
    mpd = mpd_gdf.iloc[0]
    mpd_geom = mpd.geometry
    mpd_start = mpd.start_time
    mpd_end = mpd.end_time
    
    for index, ffw in ffw_gdf.iterrows():
        # Evaluate using the exact product issuance time to catch EXTs
        prod_time = ffw.product_time
        ffw_expire = ffw.expire_time
        ffw_geom = ffw.geometry
        
        # GREEN / RED: The action (NEW or EXT) occurred DURING the MPD valid timeframe
        if mpd_start <= prod_time <= mpd_end:
            if ffw_geom.intersects(mpd_geom):
                ffw_gdf.at[index, 'plot_color'] = 'green'
            elif ffw_geom.disjoint(mpd_geom):
                ffw_gdf.at[index, 'plot_color'] = 'red'
                
        # ORANGE: The action occurred BEFORE, but is still active during any part of the MPD
        elif prod_time < mpd_start and ffw_expire >= mpd_start:
            ffw_gdf.at[index, 'plot_color'] = 'orange'
            
    # FILTER: Remove any polygons that were not classified
    ffw_gdf = ffw_gdf[ffw_gdf['plot_color'] != 'drop'].copy()
                
    # SORT: Ensure correct z-order drawing (Green on top)
    if not ffw_gdf.empty:
        priority_map = {'red': 1, 'orange': 2, 'green': 3}
        ffw_gdf['draw_priority'] = ffw_gdf['plot_color'].map(priority_map)
        ffw_gdf = ffw_gdf.sort_values(by='draw_priority').reset_index(drop=True)
                
    return ffw_gdf

def plot_mpd_verification(mpd_gdf, classified_ffw_gdf, counties_gdf, save_path=None):
    """Plots the MPD and classified FFW polygons over a customized geographic map."""
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    
    minx, miny, maxx, maxy = mpd_gdf.total_bounds
    buffer = 1.5 
    ax.set_extent([minx - buffer, maxx + buffer, miny - buffer, maxy + buffer], crs=ccrs.PlateCarree())
                  
    mpd_row = mpd_gdf.iloc[0]
    mpd_geom = mpd_row.geometry
    centroid = mpd_geom.centroid
    mpd_num = mpd_row['NUM']
    issue_date = mpd_row.start_time.strftime('%B %d, %Y')
                  
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_1_states_provinces_lines', scale='50m', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='black', linewidth=2.0, zorder=3)
    
    counties_gdf.boundary.plot(ax=ax, edgecolor='gray', linewidth=0.5, zorder=2)
    
    if not classified_ffw_gdf.empty:
        classified_ffw_gdf.plot(ax=ax, color=classified_ffw_gdf['plot_color'], edgecolor='black', alpha=0.6, zorder=4)
                                
    mpd_gdf.boundary.plot(ax=ax, color='blue', linewidth=3.0, zorder=5)
    
    plt.title(f"WPC MPD #{mpd_num} Verification - Issued: {issue_date}\nCentered on {centroid.y:.2f}N, {centroid.x:.2f}W", 
              fontsize=14, fontweight='bold')
              
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# --- 3. Dashboard Generator (Archive Version) ---

def generate_dashboard_html(target_date_str, current_images, web_root_dir):
    """Generates a static HTML dashboard with a permanent latest day and active historical archive."""
    images_dir = os.path.join(web_root_dir, "images")
    
    # 1. Scan the images folder to find all historical maps
    date_to_images = defaultdict(list)
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if filename.endswith(".png") and filename.startswith("mpd_"):
                parts = filename.split('_')
                if len(parts) >= 3:
                    date_str_raw = parts[1]
                    try:
                        date_obj = dt.strptime(date_str_raw, "%Y%m%d")
                        display_date = date_obj.strftime("%Y-%m-%d")
                        date_to_images[display_date].append(filename)
                    except ValueError:
                        continue
                        
    # 1b. Scan the root folder for past HTML files so we don't forget "quiet" days!
    if os.path.exists(web_root_dir):
        for filename in os.listdir(web_root_dir):
            if filename.endswith(".html") and filename.startswith("20"):
                # Matches format YYYY-MM-DD.html
                past_date_str = filename.replace(".html", "")
                # If this date isn't in our dictionary yet (because it had no PNGs), add it as an empty list
                if past_date_str not in date_to_images:
                    date_to_images[past_date_str] = []

    # 2. Force the CURRENT run's date into the dictionary
    date_to_images[target_date_str] = current_images
                        
    # Sort dates descending (newest at the top)
    sorted_dates = sorted(date_to_images.keys(), reverse=True)
    latest_date = sorted_dates[0]
    
    # 3. Build the navigation sidebar HTML
    nav_links_html = ""
    for d in sorted_dates:
        # The newest date ALWAYS points to index.html in the menu
        if d == latest_date:
            page_name = "index.html"
            link_text = f"{d} (Latest)"
        else:
            page_name = f"{d}.html"
            link_text = d
        nav_links_html += f'<a href="{page_name}">{link_text}</a>\n'

    # 4. Define the HTML Template
    def create_page_content(page_date, images, nav_html):
        images.sort() 
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>WPC MPD Verification Archive</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #ffffff; margin: 0; display: flex; height: 100vh; overflow: hidden; }}
                /* Sidebar Styles */
                .sidebar {{ width: 250px; background-color: #121212; padding: 20px 0; overflow-y: auto; border-right: 1px solid #333; flex-shrink: 0; }}
                .sidebar h2 {{ text-align: center; color: #4fa8fb; font-size: 1.2em; margin-top: 0; border-bottom: 1px solid #333; padding-bottom: 15px; margin-bottom: 0; }}
                .sidebar a {{ display: block; padding: 15px 20px; color: #aaaaaa; text-decoration: none; border-bottom: 1px solid #222; transition: 0.2s; }}
                .sidebar a:hover {{ background-color: #2a2a2a; color: #ffffff; }}
                .sidebar a.active-link {{ background-color: #4fa8fb; color: #121212; font-weight: bold; }}
                
                /* Main Content Styles */
                .main-content {{ flex-grow: 1; padding: 20px; overflow-y: auto; background-color: #1e1e1e; }}
                h1 {{ color: #4fa8fb; border-bottom: 1px solid #444; padding-bottom: 10px; margin-top: 0; }}
                .date-header {{ font-size: 1.2em; color: #aaaaaa; margin-bottom: 30px; font-weight: bold; }}
                .dashboard-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; max-width: 1400px; }}
                .map-card {{ background-color: #2a2a2a; padding: 15px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }}
                .map-card img {{ width: 100%; height: auto; border-radius: 4px; border: 1px solid #555; transition: 0.3s; }}
                .map-card img:hover {{ opacity: 0.8; cursor: pointer; }}
                .map-card h3 {{ margin-top: 0; font-size: 1.2em; text-align: center; color: #e0e0e0; margin-bottom: 15px; }}
                .no-data {{ font-size: 1.2em; color: #ff6b6b; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="sidebar">
                <h2>Archive Dates</h2>
                {nav_html}
            </div>
            <div class="main-content">
                <h1>WPC Mesoscale Precipitation Discussions</h1>
                <div class="date-header">Verification for: {page_date}</div>
                <div class="dashboard-grid">
        """
        
        if not images:
            html += f'<div class="no-data">No MPDs were issued on {page_date}.</div>'
        else:
            for img in images:
                mpd_num = img.split('_')[2].split('.')[0]
                html += f"""
                    <div class="map-card">
                        <h3>MPD #{mpd_num}</h3>
                        <a href="images/{img}" target="_blank" title="Click to view full resolution map">
                            <img src="images/{img}" alt="Verification Map for MPD {mpd_num}">
                        </a>
                    </div>
                """
                
        html += "</div></div></body></html>"
        return html

    # 5. Generate the HTML files
    for d in sorted_dates:
        # ALWAYS create a specific date file (e.g. 2026-03-23.html) so the script remembers it tomorrow!
        file_name_specific = f"{d}.html"
        filepath_specific = os.path.join(web_root_dir, file_name_specific)
        
        if d == latest_date:
            # This is the newest day, so highlight the index.html link
            active_nav = nav_links_html.replace('href="index.html"', 'href="index.html" class="active-link"')
            content = create_page_content(d, date_to_images[d], active_nav)
            
            # Save as the specific date file (for the archive memory)
            with open(filepath_specific, "w") as f:
                f.write(content)
                
            # AND save as index.html (for the homepage)
            index_filepath = os.path.join(web_root_dir, "index.html")
            with open(index_filepath, "w") as f:
                f.write(content)
        else:
            # Highlight the specific date link for older days
            active_nav = nav_links_html.replace(f'href="{file_name_specific}"', f'href="{file_name_specific}" class="active-link"')
            content = create_page_content(d, date_to_images[d], active_nav)
            
            with open(filepath_specific, "w") as f:
                f.write(content)
            
    print(f"Successfully generated dashboard. Archive contains {len(sorted_dates)} active dates.")

# --- 4. Main Operational Pipeline ---

def main():
    print("--- Starting WPC MPD Daily Verification Pipeline ---")
    
    target_date = datetime.datetime.now() - datetime.timedelta(days=1)
    date_str = target_date.strftime("%Y%m%d")
    html_date_str = target_date.strftime("%Y-%m-%d")

    # Set to current directory for GitHub Actions
    web_root_dir = "." 
    images_dir = os.path.join(web_root_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    mpd_gdf = fetch_iem_mpds(target_date)
    
    if mpd_gdf.empty:
        print("No MPDs issued yesterday. Generating empty dashboard.")
        generate_dashboard_html(html_date_str, [], web_root_dir)
        return

    # 00Z CROSSOVER FIX: Expand query window based on latest MPD expiration
    max_end_time = mpd_gdf['end_time'].max()
    end_date = max_end_time.to_pydatetime()
    
    ffw_gdf = fetch_iem_ffws(target_date, end_date)

    print("Loading US County boundaries...")
    county_url = "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip"
    counties_gdf = gpd.read_file(county_url)

    generated_images = []

    for index, single_mpd_series in mpd_gdf.iterrows():
        single_mpd = mpd_gdf.iloc[[index]]
        mpd_id = single_mpd.iloc[0]['NUM']
        
        print(f"Evaluating spatial and temporal overlap for MPD #{mpd_id}...")
        classified_ffws = classify_ffw_polygons(single_mpd, ffw_gdf)
        
        filename = f"mpd_{date_str}_{mpd_id}.png"
        output_filepath = os.path.join(images_dir, filename)
        
        print(f"Saving verification map for MPD #{mpd_id}...")
        plot_mpd_verification(single_mpd, classified_ffws, counties_gdf, save_path=output_filepath)
        generated_images.append(filename)

    generate_dashboard_html(html_date_str, generated_images, web_root_dir)
    print("Pipeline complete! Verification maps and dashboard are ready.")

if __name__ == "__main__":
    main()
