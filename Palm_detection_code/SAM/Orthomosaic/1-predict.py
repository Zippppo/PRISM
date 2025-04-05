"""
Object Detection Pipeline for Orthomosaic Images

This script performs object detection on large orthomosaic images using YOLO/RTDETR models.
The process includes:
1. Sliding window approach to crop large images into smaller patches
2. Object detection on each patch
3. Saving results in both CSV and YOLO format
4. Visualization of detections (optional)

Input:
    - Orthomosaic image in TIFF format
    - Pre-trained model weights
    - Model configuration

Output:
    - CSV file with detection results (including geographic coordinates)
    - YOLO format text files for each patch
    - Visualization images (optional)
"""

import ultralytics
ultralytics.checks()

import numpy as np
from ultralytics import YOLO, RTDETR
# from ultralytics import YOLOv10
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
    """Perform inference on a single image."""
    img = Image.open(image_path)
    results = model.predict(img, save=False, imgsz=800, conf=0.25, verbose=False)
    return results

def save_bounding_box_info_to_csv_and_yolo_txt(results, predictions_csv, local_txt_path, transform, crop_size, x, y):
    """
    Save detection results in both CSV and YOLO formats.
    
    Args:
        results: Detection results from model
        predictions_csv: Path to CSV file for saving results
        local_txt_path: Path to YOLO format text file
        transform: Rasterio transform for coordinate conversion
        crop_size: Size of image crop
        x, y: Crop coordinates
    """
    with open(predictions_csv, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        with open(local_txt_path, 'w') as txt_file:
            for result in results:
                if result.boxes is not None and len(result.boxes.xyxy) > 0:
                    for i, box in enumerate(result.boxes.xyxy):
                        # Extract box coordinates and convert to required formats
                        x1, y1, x2, y2 = map(float, box.tolist()[:4])
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        lon, lat = rasterio.transform.xy(transform, center_y, center_x, offset='center')
                        width = x2 - x1
                        height = y2 - y1
                        confidence = result.boxes.conf[i].item()

                        # Calculate normalized coordinates for YOLO format
                        norm_x_center = center_x / crop_size
                        norm_y_center = center_y / crop_size
                        norm_width = width / crop_size
                        norm_height = height / crop_size

                        class_id = int(result.boxes.cls[i].item())
                        class_name = result.names[class_id]

                        # Save to CSV and YOLO formats
                        csv_writer.writerow([lon, lat, width, height, class_name, x, y, center_x, center_y, confidence])
                        txt_file.write(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")

def visualize_predictions(image_path, results, save_path):
    """
    Visualize detection results on image.
    Only saves images with detections.
    
    Returns:
        bool: True if detections were found and image was saved
    """
    # Check for detections
    has_detections = False
    for result in results:
        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            has_detections = True
            break
    
    if not has_detections:
        return False
        
    img = cv2.imread(image_path)
    
    for result in results:
        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.tolist()[:4])
                confidence = result.boxes.conf[i].item()
                class_id = int(result.boxes.cls[i].item())
                class_name = result.names[class_id]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imwrite(save_path, img)
    return True

def main():
    # Get configuration from user input
    model_name = input("Enter the model name: ")
    model_path = input("Enter the model path: ")
    visualization_path = input("Enter the path to save visualization results: ")
    model = YOLO(model_path)

    # Define processing parameters
    crop_size = 800
    stride = 400
    site_number = input("Enter the site number: ")
    site_directory = f"/path/to/images/site{site_number}"
    results_directory = f"/path/to/results"

    # Create necessary directories
    os.makedirs(site_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)

    # Validate inputs and paths
    if not os.path.exists(site_directory):
        print(f"Error: Directory {site_directory} does not exist and could not be created.")
        exit()

    tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
    if not tif_files:
        print(f"Error: No TIFF files found in {site_directory}.")
        print(f"Please ensure your TIFF file is placed in the {site_directory} directory.")
        exit()

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit()

    # Process the orthomosaic image
    tif_file = tif_files[0]
    input_path = os.path.join(site_directory, tif_file)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    save_dir = os.path.join(site_directory, f'cropped_{base_name}')
    predictions_csv = os.path.join(results_directory, f'predictions_{base_name}_{model_name}.csv')

    # Make directories and CSV file
    os.makedirs(save_dir, exist_ok=True)
    with open(predictions_csv, 'w', newline='') as file:  # Overwrite mode
        writer = csv.writer(file)
        writer.writerow(['Longitude', 'Latitude', 'Width', 'Height', 'Predicted Class', 'X', 'Y', 'Xc', 'Yc', 'Conf'])

    # Open and process the full landscape image
    start_time1 = time.time()

    with rasterio.open(input_path) as src:
        image_width, image_height = src.width, src.height
        detection_count = 0  # 添加计数器
        for y in tqdm(range(0, src.height - crop_size + 1, stride), desc="Processing"):
            for x in range(0, src.width - crop_size + 1, stride):
                window = Window(x, y, crop_size, crop_size)
                transform = src.window_transform(window)
                crop_image_path = os.path.join(save_dir, f"crop_{x}_{y}.tif")
                local_txt_path = os.path.splitext(crop_image_path)[0] + '.txt'
                
                # 保存切割的图像
                with rasterio.open(crop_image_path, 'w', driver='GTiff', height=crop_size, width=crop_size, count=src.count, dtype=src.dtypes[0], transform=transform) as dst:
                    dst.write(src.read(window=window))

                # 执行预测
                results = inference_on_image(crop_image_path)
                
                # 保存预测结果到CSV和YOLO格式
                save_bounding_box_info_to_csv_and_yolo_txt(results, predictions_csv, local_txt_path, transform, crop_size, x, y)
                
                # 只保存有检测结果的图片的可视化结果
                vis_save_path = os.path.join(visualization_path, f"vis_crop_{x}_{y}.jpg")
                os.makedirs(visualization_path, exist_ok=True)
                if visualize_predictions(crop_image_path, results, vis_save_path):
                    detection_count += 1  # 如果保存了图片就增加计数

        print(f"Total images with detections: {detection_count}")

    end_time1 = time.time()
    print("Inference and result saving complete.")

    # Remove redundant predictions
    csv_path = os.path.join(site_directory, f'predictions_{base_name}_{model_name}.csv')
    output_csv_path = os.path.join(site_directory, f'filtered_{base_name}_{model_name}.csv')

    # Read the TIFF to get the transform
    with rasterio.open(input_path) as src:
        transform = src.transform

    print("Cleaning and result saving complete.")
    print(f"Total inference time: {end_time1 - start_time1} seconds")


    # Delete the folder that saved txt and tif images
    if os.path.exists(save_dir):
        # Delete all files and subdirectories in the directory
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        
        # Delete the directory itself
        shutil.rmtree(save_dir)
        print(f"Deleted folder at {save_dir}")
    else:
        print(f"Folder at {save_dir} does not exist.")


if __name__ == "__main__":
    main()