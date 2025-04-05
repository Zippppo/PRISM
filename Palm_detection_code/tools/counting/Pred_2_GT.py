"""
Description:
This script compares predicted coordinates with ground truth coordinates for multiple sites.
It computes matching statistics including match ratio, average distance, and median distance,
and generates a cumulative distribution function (CDF) plot of the distances.
For each site, the script reads a filtered prediction CSV file and a ground truth CSV file,
matches each predicted point with its nearest ground truth point within a specified distance threshold,
and saves the matching details to a CSV file.
All file paths and directories use generic placeholders to avoid exposing personal information.
"""

import os
import pandas as pd
import rasterio
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.legend_handler import HandlerLine2D, HandlerPatch

# Configuration: Update these paths with generic placeholders
IMAGES_DIR = "/path/to/your/images"      # Root directory for image data
RESULTS_DIR = "/path/to/your/results"      # Directory to store result files

def get_degree_bounds(lon, lat, meters, dataset):
    """
    Calculate the degree bounds around a specific coordinate given a distance in meters.
    
    Parameters:
        lon (float): Longitude value.
        lat (float): Latitude value.
        meters (float): Distance in meters.
        dataset: A rasterio dataset used to extract spatial resolution.
    
    Returns:
        Tuple (min_lon, max_lon, min_lat, max_lat)
    """
    pixelSizeX, pixelSizeY = dataset.transform[0], abs(dataset.transform[4])
    if pixelSizeX < 0.0001:
        meters_per_degree = 111300
        degreesX = meters / meters_per_degree
        degreesY = meters / (meters_per_degree * math.cos(math.radians(lat)))
    else:
        degreesX = meters * pixelSizeX
        degreesY = meters * pixelSizeY
    return lon - degreesX, lon + degreesX, lat - degreesY, lat + degreesY

def compare_coordinates(site_numbers, model_name, distance_threshold=10):
    """
    Compare predicted coordinates with ground truth coordinates for multiple sites.
    
    Parameters:
        site_numbers (list): A list of site numbers.
        model_name (str): The model identifier used for predictions.
        distance_threshold (float): Distance threshold in meters for matching (default is 10 meters).
    
    The function reads CSV files containing filtered predictions and ground truth data,
    finds the closest ground truth point for each predicted point within the specified threshold,
    calculates the match ratio along with mean and median distances,
    and generates a cumulative distribution function (CDF) plot with a dashed vertical line at the 90th percentile.
    The matching results are saved to CSV files.
    """
    # Reset the style to default and update plotting parameters
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    })
    
    # Create figure for the CDF plot
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Set white background for both axes and the figure
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Use a professional color palette
    site_colors = sns.color_palette("deep", len(site_numbers))
    
    legend_labels = {
        'TESORO_ESCONDIDO_2': 'Tesoro Escondido',
        'JAMACOAQUE1': 'Jama-Coaque',
        'CANANDE2': 'Canande',
        'FCAT6': 'FCAT',
    }

    for site_number, color in zip(site_numbers, site_colors):
        # Define the site directory using a generic path placeholder
        site_directory = os.path.join(IMAGES_DIR, f"site{site_number}")
        tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
        if len(tif_files) != 1:
            raise ValueError(f"There should be exactly one TIFF file in the directory for site {site_number}.")
        tif_name = tif_files[0]
        orthomosaic_file = os.path.join(site_directory, tif_name)

        filtered_csv = os.path.join(RESULTS_DIR, f'filtered_{tif_name[:-4]}_{model_name}.csv')
        tif_csv = os.path.join(RESULTS_DIR, f'{tif_name[:-4]}-GT.csv')
        filtered_df = pd.read_csv(filtered_csv)
        tif_df = pd.read_csv(tif_csv)

        with rasterio.open(orthomosaic_file) as dataset:
            closest_records = []
            count_ratio = 0
            distances = []
            for index, pred_row in filtered_df.iterrows():
                lon1, lat1 = pred_row['Longitude'], pred_row['Latitude']
                lon_min, lon_max, lat_min, lat_max = get_degree_bounds(lon1, lat1, distance_threshold, dataset)
                close_points = tif_df[
                    (tif_df['Longitude'] >= lon_min) & 
                    (tif_df['Longitude'] <= lon_max) &
                    (tif_df['Latitude'] >= lat_min) & 
                    (tif_df['Latitude'] <= lat_max)
                ].copy()
                
                if not close_points.empty:
                    count_ratio += 1
                    close_points['Distance'] = ((close_points['Longitude'] - lon1)**2 + 
                                                (close_points['Latitude'] - lat1)**2)**0.5
                    meters_per_degree = 111300
                    close_points['Distance'] *= meters_per_degree

                    closest_point = close_points.nsmallest(1, 'Distance').iloc[0]
                    closest_records.append({
                        'Predicted_Longitude': lon1,
                        'Predicted_Latitude': lat1,
                        'Human_Longitude': closest_point['Longitude'],
                        'Human_Latitude': closest_point['Latitude'],
                        'Distance': closest_point['Distance']
                    })
                    distances.append(closest_point['Distance'])

            if distances:
                # Calculate mean and median distances
                mean_distance = np.mean(distances)
                median_distance = np.median(distances)
                
                # Compute and plot the CDF of distances
                sorted_distances = np.sort(distances)
                cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
                plt.plot(sorted_distances, cdf, color=color, 
                         label=legend_labels.get(tif_name[:-4], tif_name[:-4]), 
                         linewidth=4)
                
                # Add a dashed vertical line at the 90th percentile (up to y = 0.9)
                threshold_90 = np.percentile(sorted_distances, 90)
                plt.plot([threshold_90, threshold_90], [0, 0.9], 
                         color=color, linestyle='--', alpha=0.7, linewidth=4)
            else:
                mean_distance = median_distance = None

            # Save matching results to a CSV file
            closest_df = pd.DataFrame(closest_records)
            closest_df['Mean_Distance'] = mean_distance
            closest_df['Median_Distance'] = median_distance
            out_csv = os.path.join(RESULTS_DIR, f'closest_points_pred_to_gt_{tif_name[:-4]}.csv')
            closest_df.to_csv(out_csv, index=False)
            
            # Output match ratio and distance statistics
            site_ratio = count_ratio / len(filtered_df) if len(filtered_df) > 0 else 0
            print(f"Site {site_number} (Pred reference) match ratio: {site_ratio:.4f}, "
                  f"Average distance: {mean_distance:.2f} meters, Median distance: {median_distance:.2f} meters")
    
    # Add a dummy line to include the 90% threshold in the legend
    plt.plot([], [], '--', color='gray', label='90% Threshold', linewidth=4)

    # Add a horizontal dashed line at y = 0.9
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, linewidth=2)

    # Set axis labels and limits
    plt.xlabel('Distance (meters)', fontsize=24)
    plt.ylim(0, 1.0001)
    plt.xlim(0, 6)
    
    # Set tick parameters
    ax.tick_params(width=1, color='black', labelsize=22)
    
    # Customize axis borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Remove grid lines
    ax.grid(False)
    
    # Optimize legend style
    lines = ax.get_lines()
    legend_elements = []
    used_labels = set()
    
    for line in lines:
        label = line.get_label()
        if label not in used_labels:
            if label == '90% Threshold':
                legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', 
                                                  label=label, linewidth=4))
            else:
                color = line.get_color()
                legend_elements.append(plt.Line2D([0], [0], color=color, 
                                                  label=label, linewidth=4))
            used_labels.add(label)
    
    legend = ax.legend(handles=legend_elements,
                       loc='upper right',
                       fontsize=20,
                       frameon=True,
                       edgecolor='gray',
                       fancybox=True,
                       framealpha=0.8,
                       bbox_to_anchor=(0.98, 0.98),
                       borderpad=0.7,
                       labelspacing=0.5,
                       handlelength=2.5,
                       handleheight=1)
    
    legend.get_frame().set_linewidth(1)
    
    plt.tight_layout()
    out_plot = os.path.join(RESULTS_DIR, "pred_2_gt.png")
    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Example usage for site 1
    site_numbers = [1]
    compare_coordinates(site_numbers, 'yolov10', 5)