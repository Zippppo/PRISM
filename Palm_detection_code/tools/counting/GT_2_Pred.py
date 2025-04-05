"""
Description:
This script compares ground truth coordinates against predicted coordinates for multiple sites, computes matching statistics including match ratio, mean distance, and median distance, and generates a cumulative distribution function (CDF) plot of the distances.
All inputs and outputs use generic placeholder paths to ensure that no personal or sensitive information is exposed.
Configuration:
- Update the IMAGES_DIR and RESULTS_DIR constants with appropriate generic paths.
- Ensure that file I/O addresses do not include any personal details.
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
IMAGES_DIR = "/path/to/your/images"    # Root directory for image data
RESULTS_DIR = "/path/to/your/results"   # Directory to store result files

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
    Compare ground truth coordinates with predicted coordinates for multiple sites.
    
    Parameters:
        site_numbers (list): A list of site numbers.
        model_name (str): The model identifier used for predictions.
        distance_threshold (float): Distance threshold in meters for matching (default is 10).
    
    The function reads CSV files containing filtered predictions and ground truth data,
    computes the closest predicted point for each ground truth coordinate within a specified threshold,
    calculates the match ratio along with the mean and median distances,
    and generates a CDF plot with a dashed line indicating the 90th percentile.
    The matching results are saved to CSV files.
    """
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    })
    
    # Create a new figure for plotting
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Set white background for both the axes and the figure
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

    site_labels = []

    for site_number, color in zip(site_numbers, site_colors):
        # Update the site directory using a generic placeholder path
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
            for index, true_row in tif_df.iterrows():
                lon1, lat1 = true_row['Longitude'], true_row['Latitude']
                lon_min, lon_max, lat_min, lat_max = get_degree_bounds(lon1, lat1, distance_threshold, dataset)

                close_points = filtered_df[
                    (filtered_df['Longitude'] >= lon_min) &
                    (filtered_df['Longitude'] <= lon_max) &
                    (filtered_df['Latitude'] >= lat_min) & 
                    (filtered_df['Latitude'] <= lat_max)
                ].copy()
                
                if not close_points.empty:
                    count_ratio += 1
                    close_points['Distance'] = ((close_points['Longitude'] - lon1)**2 + 
                                                (close_points['Latitude'] - lat1)**2)**0.5
                    meters_per_degree = 111300
                    close_points['Distance'] *= meters_per_degree

                    closest_point = close_points.nsmallest(1, 'Distance').iloc[0]
                    closest_records.append({
                        'Human_Longitude': lon1,
                        'Human_Latitude': lat1,
                        'Predicted_Longitude': closest_point['Longitude'],
                        'Predicted_Latitude': closest_point['Latitude'],
                        'Distance': closest_point['Distance']
                    })
                    distances.append(closest_point['Distance'])

            if distances:
                # Calculate the mean and median distances
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
                
                site_labels.append(legend_labels.get(tif_name[:-4], tif_name[:-4]))
            else:
                mean_distance = median_distance = None

            # Save the matching results to a CSV file
            closest_df = pd.DataFrame(closest_records)
            closest_df['Mean_Distance'] = mean_distance
            closest_df['Median_Distance'] = median_distance
            closest_df.to_csv(os.path.join(RESULTS_DIR, f'closest_points_gt_to_pred_{tif_name[:-4]}.csv'), 
                              index=False)
            
            # Output matching ratio and distance statistics
            site_ratio = count_ratio / len(tif_df) if len(tif_df) > 0 else 0
            print(f"Site {site_number} (GT reference) match ratio: {site_ratio:.4f}, "
                  f"Average distance: {mean_distance:.2f} meters, Median distance: {median_distance:.2f} meters")

    # Add a dummy line to represent the 90% threshold in the legend
    plt.plot([], [], '--', color='gray', label='90% Threshold', linewidth=4)

    # Add a horizontal dashed line at y = 0.9
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, linewidth=2)

    # Set axis labels and limits
    plt.xlabel('Distance (meters)', fontsize=24)
    plt.ylim(0, 1.0001)
    plt.xlim(0, 6)
    
    # Set tick parameters for the axis
    ax.tick_params(width=1, color='black', labelsize=22)
    
    # Customize axis borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Disable grid lines
    ax.grid(False)
    
    # Optimize legend style
    lines = ax.get_lines()
    legend_elements = []
    used_labels = set()
    
    for line in lines:
        label = line.get_label()
        if label not in used_labels:
            if label == '90% Threshold':
                # For '90% Threshold', use a dashed line in the legend
                legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', 
                                                    label=label, linewidth=4))
            else:
                # For other labels, use a solid line
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
                       bbox_to_anchor=(0.99, 0.98),
                       borderpad=0.7,
                       labelspacing=0.5,
                       handlelength=2.5,
                       handleheight=1)
    
    legend.get_frame().set_linewidth(1)
    
    plt.tight_layout()
    output_plot_path = os.path.join(RESULTS_DIR, "gt_2_pred.png")
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare ground truth coordinates with predicted coordinates across multiple sites."
    )
    parser.add_argument(
        "--site_numbers", type=int, nargs="+", default=[1, 2, 3, 4],
        help="List of site numbers to process (default: [1, 2, 3, 4])"
    )
    parser.add_argument(
        "--model_name", type=str, default="yolov10",
        help="Prediction model identifier (default: 'yolov10')"
    )
    parser.add_argument(
        "--distance_threshold", type=float, default=5,
        help="Distance threshold in meters for matching (default: 5)"
    )
    
    args = parser.parse_args()
    compare_coordinates(args.site_numbers, args.model_name, args.distance_threshold)