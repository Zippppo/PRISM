#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO Model Inference Script

This script performs inference on images in two directories (test and val) using pre-trained YOLO weights.
The image filenames follow the pattern 'FCAT{i}---{j}.png', where i is either 5 or 6, and j is any positive integer.

The inference results (bounding box information) are saved in JSON files, where each record includes:
    - image_id: Sequential identifier starting from 1, ordered first by i=5 then i=6, with j in ascending order
    - bbox: List in format [x, y, width, height], where (x, y) is the top-left corner
    - score: Model's confidence score
    - category_id: Fixed as 1 (single category task)

Command line arguments:
    --test_folder : Path to test images directory
    --val_folder  : Path to validation images directory
    --weights     : Path to YOLO pt weights file
    --output_dir  : Directory to save output JSON files
"""

import os
import re
import glob
import json
import argparse
import cv2
import torch
from ultralytics import YOLO

def parse_filename(filename):
    """
    Parse image filename to extract i and j values from format:
        FCAT{i}---{j}.png
    Returns tuple (i, j) if successful, None if parsing fails.
    """
    pattern = r"^FCAT(\d+)---(\d+)\.png$"
    m = re.match(pattern, filename)
    if m:
        i_val = int(m.group(1))
        j_val = int(m.group(2))
        return i_val, j_val
    return None

def sort_image_paths(image_paths):
    """
    Sort image paths based on filename pattern:
    First by i=5 then i=6, and within each group by ascending j value.
    Returns sorted list of file paths.
    """
    items = []
    for path in image_paths:
        base = os.path.basename(path)
        parsed = parse_filename(base)
        if parsed is None:
            continue
        i_val, j_val = parsed
        items.append((path, i_val, j_val))
    items_sorted = sorted(items, key=lambda x: (x[1], x[2]))
    return [item[0] for item in items_sorted]

def run_inference(model, image_path):
    """
    Perform inference on an image using YOLO model.
    
    Returns list of detection dictionaries, each containing:
        - bbox: [x, y, width, height]
        - score: confidence score
        - category_id: fixed as 1
        
    Uses OpenCV for image reading and ultralytics YOLO for inference,
    extracting prediction boxes (xyxy format) and confidence scores.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return []
    
    results = model(img)
    try:
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
    except Exception as e:
        print(f"Error getting inference results for {image_path}: {e}")
        return []
    
    bbox_list = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        width = x2 - x1
        height = y2 - y1
        bbox = [float(x1), float(y1), float(width), float(height)]
        bbox_list.append({
            "bbox": bbox,
            "score": float(confs[i]),
            "category_id": 1
        })
    return bbox_list

def process_folder(model, folder_path):
    """
    Process all PNG images in a folder: sort, perform inference, and generate predictions.
    Image IDs start from 1 and increment according to sort order.
    """
    png_files = glob.glob(os.path.join(folder_path, "*.png"))
    print(f"Found {len(png_files)} PNG files in {folder_path}")
    sorted_files = sort_image_paths(png_files)
    
    predictions = []
    image_id = 1
    for img_path in sorted_files:
        print(f"Processing image_id={image_id}, file: {img_path}")
        bboxes = run_inference(model, img_path)
        for bbox_info in bboxes:
            bbox_info["image_id"] = image_id
            predictions.append(bbox_info)
        image_id += 1
    return predictions

def main():
    parser = argparse.ArgumentParser(
        description="Perform YOLO inference on test and validation images and save results to JSON files"
    )
    parser.add_argument("--test_folder", type=str, required=True, help="Path to test images directory")
    parser.add_argument("--val_folder", type=str, required=True, help="Path to validation images directory")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO pt weights file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for JSON files")
    args = parser.parse_args()

    print("Loading model...")
    model = YOLO(args.weights)
    
    print("Processing test folder images...")
    test_predictions = process_folder(model, args.test_folder)
    os.makedirs(args.output_dir, exist_ok=True)
    test_json_path = os.path.join(args.output_dir, "test.json")
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_predictions, f, indent=4, ensure_ascii=False)
    print(f"Test results saved to: {test_json_path}")

    print("Processing validation folder images...")
    val_predictions = process_folder(model, args.val_folder)
    val_json_path = os.path.join(args.output_dir, "val.json")
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(val_predictions, f, indent=4, ensure_ascii=False)
    print(f"Validation results saved to: {val_json_path}")

if __name__ == "__main__":
    main()