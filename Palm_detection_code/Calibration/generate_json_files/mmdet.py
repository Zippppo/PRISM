"""
MMDetection Model Inference Script

This script uses MMDetection framework to perform inference on images in two subdirectories: 'val' and 'test'.
The image filenames follow the pattern "FCAT{i}---{j}.png", where `i` is either 5 or 6, and `j` is any positive integer.

The inference results (bounding box information) are saved in JSON files, where each record includes:
    - image_id: Sequential identifier starting from 1, ordered first for images with i=5 (sorted by j in ascending order)
                and then for images with i=6 (sorted by j in ascending order).
    - bbox: List in format [x, y, width, height], where (x, y) represents the top-left coordinate of the bounding box,
            and width and height are its dimensions.
    - score: Model's confidence score
    - category_id: Fixed as 1 (single category task)

Command line arguments:
    --cfg_path   : Path to the model configuration file
    --checkpoint : Path to the model weights file
    --img_dir    : Root directory containing images (with 'val' and 'test' subdirectories)
    --output_dir : Directory to save output JSON files
"""

import mmcv
from mmdet.apis import init_detector, inference_detector
import os.path as osp
import json
import numpy as np
from tqdm import tqdm
import os
import argparse

def convert_to_xywh(bbox):
    """
    Convert a bounding box from [x1, y1, x2, y2] format to [x, y, width, height] format.
    
    Args:
        bbox (list or array): Bounding box in [x1, y1, x2, y2] format.
        
    Returns:
        list: Bounding box in [x, y, width, height] format.
    """
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]

def get_image_numbers(filename):
    """
    Extract numbers from the filename for sorting.
    
    Args:
        filename (str): Filename in the format 'FCAT{i}---{j}.png'.
        
    Returns:
        tuple: A tuple (i, j) used for sorting.
    """
    name = filename.replace('.png', '')
    i, j = name.replace('FCAT', '').split('---')
    return (int(i), int(j))

def inference_and_save(config_file, checkpoint_file, img_dir, out_file):
    """
    Perform inference on images in the specified directory and save the results to a JSON file.
    
    Args:
        config_file (str): Path to the model configuration file.
        checkpoint_file (str): Path to the model weights file.
        img_dir (str): Directory path containing the images.
        out_file (str): Output JSON file path.
    """
    # Initialize the model
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    results = []
    score_thr = 0.001  # Confidence score threshold
    
    # Retrieve and sort image files
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    img_files.sort(key=get_image_numbers)
    
    # Perform inference on each image
    for img_id, img_file in enumerate(tqdm(img_files, desc="Processing images"), 1):
        img_path = osp.join(img_dir, img_file)
        result = inference_detector(model, img_path)
        
        # Extract detection results
        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        
        # Convert and store detection results
        for bbox, score in zip(bboxes, scores):
            if score > score_thr:
                bbox_xywh = [float(x) for x in convert_to_xywh(bbox)]
                det_result = {
                    "image_id": img_id,
                    "bbox": bbox_xywh,
                    "score": float(score),
                    "category_id": 1
                }
                results.append(det_result)
    
    # Sort results by confidence score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Create output directory if needed
    out_dir = osp.dirname(out_file)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    
    # Save results to JSON file
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(
        description="Command-line inference and saving results to JSON files for 'val' and 'test' datasets."
    )
    parser.add_argument(
        '--cfg_path',
        required=True,
        help='Path to the model configuration file.'
    )
    parser.add_argument(
        '--pth_path',
        required=True,
        help='Path to the model weights file.'
    )
    parser.add_argument(
        '--img_dir',
        required=True,
        help='Root directory of images containing "val" and "test" subdirectories.'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory where the output JSON files will be saved.'
    )
    args = parser.parse_args()
    
    config_file = args.cfg_path
    pth_file = args.pth_path
    
    # Process validation and test sets
    val_img_dir = osp.join(args.img_dir, "val")
    test_img_dir = osp.join(args.img_dir, "test")
    
    val_out_file = osp.join(args.output_dir, "val.json")
    test_out_file = osp.join(args.output_dir, "test.json")
    
    print("Processing validation set...")
    inference_and_save(config_file, pth_file, val_img_dir, val_out_file)
    
    print("Processing test set...")
    inference_and_save(config_file, pth_file, test_img_dir, test_out_file)

if __name__ == '__main__':
    main()