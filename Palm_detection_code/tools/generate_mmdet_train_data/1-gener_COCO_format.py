#!/usr/bin/env python
"""
This script converts JSON files from an original custom format into the COCO dataset format,
and copies the corresponding PNG images to a target directory.

The conversion includes:
  1. Extracting 'imagePath', 'imageHeight', and 'imageWidth' from the original JSON data to populate the 'images' field.
  2. Processing the 'shapes' array to convert each rectangle into a COCO annotation (calculating bbox, area, etc.).
  3. Setting all annotation 'category_id' values to 1, with the 'categories' field containing only one category "palm".

Note:
  - Replace file paths and any personal information (e.g., file directories, contributor names) with generic placeholders.
  - The original code logic remains unchanged.
"""

import os
import json
import argparse
import datetime
import shutil

def convert_original_to_coco(orig_data):
    """
    Convert the original JSON data format to COCO format.
    
    Changes include:
      1. Extracting 'imagePath', 'imageHeight', and 'imageWidth' from orig_data for the images field.
      2. Iterating over the 'shapes' array and converting each rectangle into a COCO annotation (calculating bbox, area, etc.).
      3. Setting all annotations' 'category_id' to 1 and using a single category "palm" in the categories field.
    """
    # Construct the 'info' field
    today = datetime.date.today().strftime("%Y/%m/%d")
    coco_data = {
        "info": {
            "description": "Converted to COCO format",
            "version": "1.0",
            "year": datetime.date.today().year,
            "contributor": "anonymous",
            "date_created": today
        },
        "licenses": []
    }
    
    # Construct the 'images' field (each JSON file corresponds to one image)
    img_file = orig_data.get("imagePath", "")
    img_height = orig_data.get("imageHeight", 0)
    img_width = orig_data.get("imageWidth", 0)
    coco_data["images"] = [{
        "id": 1,
        "file_name": img_file,
        "height": img_height,
        "width": img_width
    }]
    
    # Construct the 'annotations' field by converting the 'shapes' array
    annotations = []
    anno_id = 1
    for shape in orig_data.get("shapes", []):
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        # Calculate the top-left and bottom-right corners to ensure correct coordinates
        x0, y0 = points[0]
        x1, y1 = points[1]
        x_min = min(x0, x1)
        y_min = min(y0, y1)
        x_max = max(x0, x1)
        y_max = max(y0, y1)
        width = x_max - x_min
        height = y_max - y_min
        annotation = {
            "id": anno_id,
            "image_id": 1,
            "category_id": 1,  # Fixed to 1
            "bbox": [x_min, y_min, width, height],
            "area": width * height,
            "iscrowd": 0
        }
        annotations.append(annotation)
        anno_id += 1
    
    coco_data["annotations"] = annotations
    
    # Set the 'categories' field with only one category "palm"
    coco_data["categories"] = [{
        "id": 1,
        "name": "palm"
    }]
    
    return coco_data

def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON files from a source directory to COCO format and copy PNG images to the target directory."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="/path/to/your/source_directory",
        help="Directory containing the original JSON files and images."
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="/path/to/your/destination_directory",
        help="Directory to save the converted COCO JSON files and images."
    )
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # First, convert all JSON files
    for file_name in os.listdir(src_dir):
        if file_name.lower().endswith(".json"):
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            with open(src_path, "r", encoding="utf-8") as fin:
                orig_data = json.load(fin)
            coco_data = convert_original_to_coco(orig_data)
            with open(dst_path, "w", encoding="utf-8") as fout:
                json.dump(coco_data, fout, ensure_ascii=False, indent=4)
            print(f"Converted: {src_path} -> {dst_path}")
    
    # Then, copy all PNG images from the source directory to the target directory
    for file_name in os.listdir(src_dir):
        if file_name.lower().endswith(".png"):
            src_img_path = os.path.join(src_dir, file_name)
            dst_img_path = os.path.join(dst_dir, file_name)
            # Copy the image while retaining the original file
            shutil.copy(src_img_path, dst_img_path)
            print(f"Copied image: {src_img_path} -> {dst_img_path}")

if __name__ == "__main__":
    main()