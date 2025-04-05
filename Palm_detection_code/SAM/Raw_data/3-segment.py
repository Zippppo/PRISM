"""
Description:
This script applies the Segment Anything Model (SAM) to generate segmentation masks for detected objects in raw images.
It processes filtered detection results from a prior detection phase and creates visualization overlays.
The script supports multiple SAM model variants, utilizes window-based processing for large images,
and computes processing statistics for batch processing.
Configuration:
- Modify the model paths, input images folder, JSON folder, and output directories accordingly.
- No personal information is contained; placeholder paths are used for file I/O.

Segment Anything Model (SAM) Application for Raw Images

This script applies SAM to generate segmentation masks for detected objects in raw images.
It processes the filtered detection results from the NMS phase.

Features:
- Supports multiple SAM variants (mobile_sam, sam_b, sam2_b)
- Handles large images through window-based processing
- Generates visualization overlays
- Processes multiple images in batch

Input:
    - Original JPG images
    - Filtered detection results (JSON)
    - SAM model selection
Output:
    - Segmentation mask overlays
    - Processing statistics
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import torch
import time
from tqdm import tqdm
import json
import cv2
from ultralytics import SAM

# Configuration constants (modify these paths as necessary)
MODEL_FOLDER = "/path/to/your/model"
OUTPUT_FOLDER = "/path/to/your/output"
SEGMENTATION_OUTPUT_FOLDER = "/path/to/your/output/segmentation_results"

def apply_segmentation_to_image(image_path, json_path, model_choice):
    """
    Apply segmentation to a single image using the SAM model.
    
    Parameters:
        image_path (str): Path to the input image.
        json_path (str): Path to the JSON file containing detection results.
        model_choice (int): Model selection (1 for mobile_sam, 2 for sam_b, 3 for sam2_b).
        
    This function saves the segmentation overlay to a designated output folder.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    # Check if the JSON file exists
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        print("Please run the prediction script first to generate the predictions JSON file.")
        return

    # Load detection results
    with open(json_path, 'r') as f:
        detection_data = json.load(f)
    
    # Read image and convert from BGR to RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    
    # Create a global mask for the entire image
    mask_global = np.zeros((img_height, img_width), dtype=np.uint8)

    # Select SAM model based on the provided choice
    model_files = {
        1: os.path.join(MODEL_FOLDER, 'mobile_sam.pt'),
        2: os.path.join(MODEL_FOLDER, 'sam_b.pt'),
        3: os.path.join(MODEL_FOLDER, 'sam2_b.pt')
    }
    model_prefix = {1: 'SAMm', 2: 'SAM', 3: 'SAM2'}
    model_file = model_files[model_choice]
    
    # Load SAM model
    model = SAM(model_file)
    
    print("Processing detections...")
    
    # Process detections for each window in the detection data
    for window_key, detections in tqdm(detection_data['windows'].items()):
        window_x, window_y = map(int, window_key.split('_'))
        
        for detection in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection['bbox']
            
            # Convert to global coordinates
            global_x1 = int(x1 + window_x)
            global_y1 = int(y1 + window_y)
            global_x2 = int(x2 + window_x)
            global_y2 = int(y2 + window_y)
            
            # Extract the ROI region (800x800)
            roi_x1 = max(global_x1 - 400, 0)
            roi_y1 = max(global_y1 - 400, 0)
            roi_x2 = min(roi_x1 + 800, img_width)
            roi_y2 = min(roi_y1 + 800, img_height)
            
            roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Convert bounding box coordinates to the ROI coordinate system
            box_x1 = global_x1 - roi_x1
            box_y1 = global_y1 - roi_y1
            box_x2 = global_x2 - roi_x1
            box_y2 = global_y2 - roi_y1
            
            # Process segmentation using SAM with a PIL image as input
            roi_pil = Image.fromarray(roi)
            results = model(roi_pil, bboxes=[[box_x1, box_y1, box_x2, box_y2]], verbose=False)
            
            # Process segmentation results
            for r in results:
                masks = r.masks.data
                if masks is not None and len(masks) > 0:
                    mask_combined = torch.any(masks, dim=0).int()
                    mask_combined = mask_combined.cpu().numpy()
                    # Copy the segmentation result into the global mask
                    mask_global[roi_y1:roi_y2, roi_x1:roi_x2] = np.maximum(
                        mask_global[roi_y1:roi_y2, roi_x1:roi_x2],
                        mask_combined
                    )

    # Create visualization overlay using a custom colormap
    colors = ['black', 'red']
    cmap = ListedColormap(colors)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_name = f"{model_prefix[model_choice]}_{base_name}.png"
    # Save the output to the designated output folder
    output_image_path = os.path.join(OUTPUT_FOLDER, output_image_name)

    plt.figure(figsize=(img_width/100, img_height/100), dpi=100)
    plt.imshow(img)
    plt.imshow(mask_global, cmap=cmap, alpha=0.5)
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Segmentation overlay saved to {output_image_path}")

def process_single_image(image_path, json_path, model, model_choice):
    """
    Process segmentation for a single image.
    
    Parameters:
        image_path (str): Path to the input image.
        json_path (str): Path to the JSON file containing detection results.
        model: The loaded SAM model.
        model_choice (int): Model selection (1 for mobile_sam, 2 for sam_b, 3 for sam2_b).
        
    Returns:
        processing_time (float): The time (in seconds) taken for the segmentation process.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    # Check if the JSON file exists
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return None
    
    # Load image and detection results
    with open(json_path, 'r') as f:
        detection_data = json.load(f)
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    
    # Create a global mask for the entire image
    mask_global = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Start timing the segmentation process
    start_time = time.time()
    
    # Process detections for each window in the detection data
    for window_key, detections in tqdm(detection_data['windows'].items(), 
                                       desc=f"Processing {os.path.basename(image_path)}", 
                                       leave=False):
        window_x, window_y = map(int, window_key.split('_'))
        
        for detection in detections:
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection['bbox']
            
            # Convert to global coordinates
            global_x1 = int(x1 + window_x)
            global_y1 = int(y1 + window_y)
            global_x2 = int(x2 + window_x)
            global_y2 = int(y2 + window_y)
            
            # Extract the ROI region
            roi_x1 = max(global_x1 - 400, 0)
            roi_y1 = max(global_y1 - 400, 0)
            roi_x2 = min(roi_x1 + 800, img_width)
            roi_y2 = min(roi_y1 + 800, img_height)
            
            roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Convert bounding box coordinates to the ROI coordinate system
            box_x1 = global_x1 - roi_x1
            box_y1 = global_y1 - roi_y1
            box_x2 = global_x2 - roi_x1
            box_y2 = global_y2 - roi_y1
            
            # Process segmentation using SAM with a PIL image as input
            roi_pil = Image.fromarray(roi)
            results = model(roi_pil, bboxes=[[box_x1, box_y1, box_x2, box_y2]], verbose=False)
            
            # Process segmentation results
            for r in results:
                masks = r.masks.data
                if masks is not None and len(masks) > 0:
                    mask_combined = torch.any(masks, dim=0).int()
                    mask_combined = mask_combined.cpu().numpy()
                    mask_global[roi_y1:roi_y2, roi_x1:roi_x2] = np.maximum(
                        mask_global[roi_y1:roi_y2, roi_x1:roi_x2],
                        mask_combined
                    )
    
    # End timing the segmentation process
    processing_time = time.time() - start_time
    
    # Prepare the output file path using the generic segmentation output folder
    model_prefix = {1: 'SAMm', 2: 'SAM', 3: 'SAM2'}
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_name = f"{model_prefix[model_choice]}_{base_name}.png"
    output_image_path = os.path.join(SEGMENTATION_OUTPUT_FOLDER, output_image_name)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    
    plt.figure(figsize=(img_width/100, img_height/100), dpi=100)
    plt.imshow(img)
    plt.imshow(mask_global, cmap=ListedColormap(['black', 'red']), alpha=0.5)
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return processing_time

def main():
    """
    Main function to process a batch of images.
    
    It prompts for the folder paths for JPG images and JSON files,
    selects the SAM model variant, and computes segmentation processing statistics.
    """
    # Prompt for the images folder path (modify the example path as needed)
    images_folder = input("Enter the path to your JPG images folder (e.g., /path/to/your/jpg/images): ")
    # Prompt for the JSON folder path (modify the example path as needed)
    json_folder = input("Enter the path to your JSON files folder (e.g., /path/to/your/json): ")
    model_choice = int(input("Enter the model number (1 for mobile_sam, 2 for sam_b, 3 for sam2_b): "))
    
    # Validate input folders
    if not os.path.exists(images_folder):
        print(f"Error: Images folder {images_folder} does not exist.")
        return
    if not os.path.exists(json_folder):
        print(f"Error: JSON folder {json_folder} does not exist.")
        return
    
    # Load SAM model using the generic model folder
    model_files = {
        1: os.path.join(MODEL_FOLDER, 'mobile_sam.pt'),
        2: os.path.join(MODEL_FOLDER, 'sam_b.pt'),
        3: os.path.join(MODEL_FOLDER, 'sam2_b.pt')
    }
    model = SAM(model_files[model_choice])
    
    # Get all JPG files in the images folder
    jpg_files = [f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')]
    processing_times = []
    
    print(f"Found {len(jpg_files)} images to process")
    
    # Process each image file
    for jpg_file in tqdm(jpg_files, desc="Overall progress"):
        image_path = os.path.join(images_folder, jpg_file)
        base_name = os.path.splitext(jpg_file)[0]
        json_path = os.path.join(json_folder, f'detections_{base_name}_yolov10.json')
        
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found for {jpg_file}, skipping...")
            continue
        
        try:
            processing_time = process_single_image(image_path, json_path, model, model_choice)
            if processing_time is not None:
                processing_times.append(processing_time)
        except Exception as e:
            print(f"Error processing {jpg_file}: {str(e)}")
            continue
    
    # Compute and display processing statistics
    if processing_times:
        avg_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        print("\nProcessing statistics:")
        print(f"Average time per image: {avg_time:.2f} seconds")
        print(f"Standard deviation: {std_time:.2f} seconds")
        print(f"Total images processed: {len(processing_times)}")
        print(f"Total processing time: {sum(processing_times):.2f} seconds")
    else:
        print("No images were successfully processed")

if __name__ == "__main__":
    main()
