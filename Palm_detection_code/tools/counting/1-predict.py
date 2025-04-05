"""
Description:
This script performs object detection inference on large TIFF images by splitting them into smaller crops,
running predictions with a YOLO model, saving detection results in CSV and YOLO-format TXT files,
and generating visualization images for crops with detections.
It is designed for processing remote sensing imagery and cleans up intermediate files after processing.

Configuration:
- Update the file paths for the model, input TIFF images, results, and visualization outputs with appropriate generic placeholders.
- Ensure that no personal or sensitive information is included in these paths.
"""

import ultralytics
ultralytics.checks()

import numpy as np
from ultralytics import YOLO, RTDETR
import cv2
import time
import os
import csv
import rasterio
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm
import pandas as pd
from shapely.geometry import box
import glob
import shutil

def inference_on_image(image_path):
    """
    Perform inference on a single image.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        results: The detection results from the model.
    """
    img = Image.open(image_path)
    results = model.predict(img, save=False, imgsz=800, conf=0.25, verbose=False)
    return results

def save_bounding_box_info_to_csv_and_yolo_txt(results, predictions_csv, local_txt_path, transform, crop_size, x, y):
    """
    Save bounding box information into a CSV file and YOLO-format TXT file.

    Parameters:
        results: Detection results from the model.
        predictions_csv (str): Path to the CSV file for appending prediction details.
        local_txt_path (str): Path to the TXT file for YOLO-format annotations.
        transform: The affine transform for the current crop.
        crop_size (int): The size of the crop.
        x (int): The x-coordinate of the crop origin.
        y (int): The y-coordinate of the crop origin.
    """
    with open(predictions_csv, mode='a', newline='') as csv_file:  # Append mode
        csv_writer = csv.writer(csv_file)

        with open(local_txt_path, 'w') as txt_file:
            for result in results:
                if result.boxes is not None and len(result.boxes.xyxy) > 0:
                    for i, box in enumerate(result.boxes.xyxy):
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(float, box.tolist()[:4])
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        lon, lat = rasterio.transform.xy(transform, center_y, center_x, offset='center')
                        width = x2 - x1
                        height = y2 - y1
                        confidence = result.boxes.conf[i].item()

                        # Normalized coordinates in YOLO format
                        norm_x_center = center_x / crop_size
                        norm_y_center = center_y / crop_size
                        norm_width = width / crop_size
                        norm_height = height / crop_size

                        # Class information
                        class_id = int(result.boxes.cls[i].item())
                        class_name = result.names[class_id]

                        # Append the prediction information to the CSV file
                        csv_writer.writerow([lon, lat, width, height, class_name, x, y, center_x, center_y, confidence])

                        # Write YOLO-formatted annotation to the TXT file
                        txt_file.write(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")

def visualize_predictions(image_path, results, save_path):
    """
    Visualize detection results on the image and save the visualization if detections exist.

    Parameters:
        image_path (str): Path to the input image.
        results: The detection results from the model.
        save_path (str): Path where the visualization image is saved.

    Returns:
        bool: True if at least one detection exists and the image is saved, False otherwise.
    """
    # Check if any detection exists
    has_detections = False
    for result in results:
        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            has_detections = True
            break
    
    # If no detections, return immediately
    if not has_detections:
        return False
        
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Draw each detection bounding box on the image
    for result in results:
        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            for i, box in enumerate(result.boxes.xyxy):
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.tolist()[:4])
                confidence = result.boxes.conf[i].item()
                class_id = int(result.boxes.cls[i].item())
                class_name = result.names[class_id]
                
                # Draw a red rectangle for the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add the class label and confidence score
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save the visualization image
    cv2.imwrite(save_path, img)
    return True

# File-level configuration: update these paths with generic placeholders.
# Prompt the user to input the model name, model path, and visualization results directory.
model_name = input("Enter the model name: ")
model_path = input("Enter the model path (e.g., /path/to/your/model.pt): ")
visualization_path = input("Enter the path to save visualization results (e.g., /path/to/your/visualization/results): ")
model = YOLO(model_path)

# Define parameters for image cropping and detection
crop_size = 800
stride = 400
site_number = input("Enter the site number: ")
# Update site directory with a generic placeholder path for TIFF images
site_directory = os.path.join("/path/to/your/tif/images", f"site{site_number}")
# Update results directory with a generic placeholder path for storing results
results_directory = "/path/to/your/results/directory"

# Ensure that necessary directories exist
os.makedirs(site_directory, exist_ok=True)
os.makedirs(results_directory, exist_ok=True)

# Validate the existence of the TIFF images directory
if not os.path.exists(site_directory):
    print(f"Error: Directory {site_directory} does not exist and could not be created.")
    exit()

# Get all TIFF files from the site directory
tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
if not tif_files:
    print(f"Error: No TIFF files found in {site_directory}.")
    print(f"Please ensure your TIFF file is placed in the {site_directory} directory.")
    exit()

# Validate the model file path
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

tif_file = tif_files[0]
input_path = os.path.join(site_directory, tif_file)
base_name = os.path.splitext(os.path.basename(input_path))[0]
save_dir = os.path.join(site_directory, f'cropped_{base_name}')
predictions_csv = os.path.join(results_directory, f'predictions_{base_name}_{model_name}.csv')

# Create the cropped images folder and initialize the predictions CSV file (overwrite mode)
os.makedirs(save_dir, exist_ok=True)
with open(predictions_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Longitude', 'Latitude', 'Width', 'Height', 'Predicted Class', 'X', 'Y', 'Xc', 'Yc', 'Conf'])

# Begin processing the full TIFF image
start_time1 = time.time()

with rasterio.open(input_path) as src:
    image_width, image_height = src.width, src.height
    detection_count = 0  # Counter for the number of crops with at least one detection
    for y in tqdm(range(0, src.height - crop_size + 1, stride), desc="Processing"):
        for x in range(0, src.width - crop_size + 1, stride):
            window = Window(x, y, crop_size, crop_size)
            transform = src.window_transform(window)
            crop_image_path = os.path.join(save_dir, f"crop_{x}_{y}.tif")
            local_txt_path = os.path.splitext(crop_image_path)[0] + '.txt'
            
            # Save the cropped image
            with rasterio.open(crop_image_path, 'w', driver='GTiff', height=crop_size, width=crop_size,
                               count=src.count, dtype=src.dtypes[0], transform=transform) as dst:
                dst.write(src.read(window=window))

            # Perform detection on the cropped image
            results = inference_on_image(crop_image_path)
            
            # Save detection results to CSV and YOLO-format TXT
            save_bounding_box_info_to_csv_and_yolo_txt(results, predictions_csv, local_txt_path, transform, crop_size, x, y)
            
            # Save visualization image only if detections are found
            vis_save_path = os.path.join(visualization_path, f"vis_crop_{x}_{y}.jpg")
            os.makedirs(visualization_path, exist_ok=True)
            if visualize_predictions(crop_image_path, results, vis_save_path):
                detection_count += 1  # Increase counter if the image with detections is saved

    print(f"Total images with detections: {detection_count}")

end_time1 = time.time()
print("Inference and result saving complete.")

# Clean up intermediate files and directories
csv_path = os.path.join(site_directory, f'predictions_{base_name}_{model_name}.csv')
output_csv_path = os.path.join(site_directory, f'filtered_{base_name}_{model_name}.csv')

# Read the TIFF file to obtain the affine transform
with rasterio.open(input_path) as src:
    transform = src.transform

print("Cleaning and result saving complete.")
print(f"Total inference time: {end_time1 - start_time1} seconds")

# Delete the directory that stores the cropped TIFF and TXT files
if os.path.exists(save_dir):
    # Remove all files and subdirectories within the directory
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
    # Remove the directory itself
    shutil.rmtree(save_dir)
    print(f"Deleted folder at {save_dir}")
else:
    print(f"Folder at {save_dir} does not exist.")
