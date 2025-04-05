"""
Object Detection Pipeline for Raw Images

This script performs object detection on raw images using YOLO models.
Unlike the orthomosaic pipeline, this processes regular images by:
1. Dividing large images into smaller patches
2. Performing batch inference on patches
3. Saving results in JSON format with window coordinates

Input:
    - Raw images in JPG format
    - Pre-trained YOLO model weights
    - Model configuration

Output:
    - JSON file containing detection results for each image, including:
        - Window coordinates
        - Bounding boxes
        - Confidence scores
        - Class information
"""

import ultralytics
ultralytics.checks()

import numpy as np
from ultralytics import YOLO
import cv2
import time
import os
import json
from PIL import Image
from tqdm import tqdm
import shutil

# Global constants
CROP_SIZE = 800
STRIDE = 400
RESULTS_DIRECTORY = "/path/to/results"

def inference_on_image(model, image_path):
    """
    Perform inference on a single image.
    
    Args:
        model: YOLO model instance
        image_path: Path to input image
        
    Returns:
        Detection results from YOLO model
    """
    img = Image.open(image_path)
    results = model.predict(img, save=False, imgsz=800, conf=0.25, verbose=False)
    return results

def save_detection_results(results, detections_dict, window_x, window_y):
    """
    Save detection results to dictionary with window coordinates.
    
    Args:
        results: Detection results from model
        detections_dict: Dictionary to store results
        window_x: X coordinate of current window
        window_y: Y coordinate of current window
    """
    window_key = f"{window_x}_{window_y}"
    detections_dict[window_key] = []
    
    for result in results:
        if result.boxes is not None and len(result.boxes.xyxy) > 0:
            for i, box in enumerate(result.boxes.xyxy):
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(float, box.tolist()[:4])
                confidence = float(result.boxes.conf[i].item())
                class_id = int(result.boxes.cls[i].item())
                class_name = result.names[class_id]
                
                # Save detection result
                detection = {
                    'window_coords': [window_x, window_y],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }
                detections_dict[window_key].append(detection)

def process_single_image(image_path, model, model_name):
    """
    Process a single image using sliding window approach.
    
    Args:
        image_path: Path to input image
        model: YOLO model instance
        model_name: Name of the model for output file naming
        
    Returns:
        Path to results JSON file
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(os.path.dirname(image_path), f'cropped_{base_name}')
    results_path = os.path.join(RESULTS_DIRECTORY, f'detections_{base_name}_{model_name}.json')
    
    # Create temporary directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize detection results dictionary
    all_detections = {
        'image_name': os.path.basename(image_path),
        'model_name': model_name,
        'windows': {}
    }
    
    # Read and process image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    
    # Store all image patches
    crops = []
    crop_coords = []
    
    # Sliding window processing
    for y in range(0, img_height - CROP_SIZE + 1, STRIDE):
        for x in range(0, img_width - CROP_SIZE + 1, STRIDE):
            crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
            crops.append(crop)
            crop_coords.append((x, y))
    
    # Batch prediction
    batch_size = 22
    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i+batch_size]
        batch_results = model.predict(batch_crops, save=False, imgsz=800, conf=0.25, verbose=False)
        
        # Save detection results for each batch
        for j, results in enumerate(batch_results):
            x, y = crop_coords[i + j]
            save_detection_results(results, all_detections['windows'], x, y)
    
    # Save results to JSON
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_detections, f, indent=4)
    
    # Clean up temporary files
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    return results_path

def main():
    """Main execution function"""
    try:
        # Get model information
        model_name = input("Enter the model name: ")
        model_path = input("Enter the model path: ")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = YOLO(model_path)
        
        # Get input folder path
        input_folder = input("Enter the folder path containing JPG images: ")
        os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
        
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        # Get all jpg files
        jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
        if not jpg_files:
            raise ValueError(f"No JPG files found in {input_folder}")
        
        print(f"Found {len(jpg_files)} JPG files")
        
        # Process each image
        for jpg_file in tqdm(jpg_files, desc="Processing images"):
            image_path = os.path.join(input_folder, jpg_file)
            try:
                process_single_image(image_path, model, model_name)
            except Exception as e:
                print(f"Error processing {jpg_file}: {str(e)}")
                continue
        
        print("All images processed successfully")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())