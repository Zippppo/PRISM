"""
Segment Anything Model (SAM) Application

This script applies SAM to generate segmentation masks for detected objects.
It uses the filtered detection results from the NMS phase and produces segmentation overlays.

Features:
- Supports multiple SAM variants (mobile_sam, sam_b, sam2_b)
- Processes large orthomosaic images
- Generates visualization overlays
- Handles geographic coordinates
"""

import os
import rasterio
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import torch
import time
from tqdm import tqdm

from ultralytics import SAM

def apply_segmentation_to_landscape(site_number, model_name, model_choice):
    site_directory = f"images/site{site_number}"
    results_directory = "v10results"
    
    tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
    if not tif_files:
        print(f"No TIFF files found in {site_directory}.")
        return
    tif_file = tif_files[0]
    input_tif_path = os.path.join(site_directory, tif_file)
    base_name = os.path.splitext(tif_file)[0]
    
    csv_path = os.path.join(results_directory, f'filtered_{base_name}_{model_name}.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run predict.py first to generate the predictions CSV file.")
        return
        
    with rasterio.open(input_tif_path) as src:
        data = src.read([1, 2, 3])
        transform = src.transform
        mask_global = np.zeros((src.height, src.width), dtype=np.uint8)

    df = pd.read_csv(csv_path)
    df['y_center'], df['x_center'] = zip(*df.apply(lambda row: rasterio.transform.rowcol(transform, row['Longitude'], row['Latitude']), axis=1))
    df['x1'] = df['x_center'] - df['Width'] / 2
    df['y1'] = df['y_center'] - df['Height'] / 2
    df['x2'] = df['x_center'] + df['Width'] / 2
    df['y2'] = df['y_center'] + df['Height'] / 2
    df['class_id'] = 1

    model_files = {1: 'mobile_sam.pt', 2: 'sam_b.pt', 3: 'sam2_b.pt'}
    model_prefix = {1: 'SAMm', 2: 'SAM', 3: 'SAM2'}
    model_file = model_files[model_choice]

    model = SAM(model_file) # mobile_sam sam_l sam_b

    print(f"Processing {len(df)} detections...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing segments"):
        x, y, width, height = int(row['x_center']), int(row['y_center']), int(row['Width']), int(row['Height'])
        x1, y1 = max(x - 400, 0), max(y - 400, 0)
        x2, y2 = min(x1 + 800, src.width), min(y1 + 800, src.height)
        x1, y1 = x2 - 800, y2 - 800

        cropped_image = data[:, y1:y2, x1:x2]
        cropped_image = np.moveaxis(cropped_image, 0, -1)
        image_pil = Image.fromarray(np.uint8(cropped_image * 255))

        results = model(image_pil, bboxes=[[400 - width // 2, 400 - height // 2, width // 2 + 400, height // 2 + 400]], verbose=False)

        for r in results:
            masks = r.masks.data
            if masks is not None and len(masks) > 0:
                mask_combined = torch.any(masks, dim=0).int()
                mask_combined = mask_combined.cpu().numpy()
                mask_global[y1:y2, x1:x2] = np.maximum(mask_global[y1:y2, x1:x2], mask_combined)

    colors = ['black', 'red']  # 只使用背景色和红色
    cmap = ListedColormap(colors)

    print(f"Mask shape: {mask_global.shape}")
    print(f"Unique values in mask: {np.unique(mask_global)}")

    output_image_name = f"{model_prefix[model_choice]}_{base_name}.png"
    output_image_path = os.path.join(site_directory, output_image_name)

    plt.figure(figsize=(data.shape[2] / 100, data.shape[1] / 100), dpi=100)
    plt.imshow(np.moveaxis(data, 0, -1))
    plt.imshow(mask_global, cmap=cmap, alpha=0.5)
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Segmentation overlay saved to {output_image_path}")

site_number = input("Enter the site number: ")
model_name = input("Enter the model name: ")
model_choice = int(input("Enter the model number (1 for mobile_sam, 2 for sam_b, 3 for sam2_b): "))

start_time = time.time()
apply_segmentation_to_landscape(site_number, model_name, model_choice)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds.")
