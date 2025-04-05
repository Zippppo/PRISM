"""
This module provides functionality for preparing YOLO-format training datasets and generating
ground truth (GT) files in COCO format, as well as performing post-training predictions using a YOLO model.

Components:
1. PreTrainingProcessor:
   - Prepares multiple YOLO training datasets from a source dataset that contains JSON annotations and images.
   - Splits the dataset into training, validation, and testing sets based on provided split ratios.
   - Converts annotation coordinates to YOLO format and copies the corresponding images.
   - Generates GT files in COCO format for validation and testing sets.

2. PostTrainingProcessor:
   - Performs post-training processing by using a trained YOLO model for inference on validation and test datasets.
   - Saves prediction results in COCO format.

Note:
  - All file paths have been replaced with generic placeholders (e.g., '/path/to/your/...').
  - The original code logic is preserved.
"""

import os
import json
import shutil
import random
import numpy as np
from PIL import Image
import torch
from pathlib import Path

class PreTrainingProcessor:
    def __init__(self, config):
        """
        Pre-training configuration.

        Args:
            config (dict): Configuration dictionary with the following keys:
                - source_dir: Path to the original dataset containing JSON annotations and images.
                - dataset_output_dir: Output path for the YOLO-format dataset.
                - gt_output_dir: Output directory for the GT files.
                - split_ratio: Dataset split ratios, e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15}.
                - num_datasets: Number of datasets to generate (default is 5).
                - random_seed: Random seed (default is 42).
        """
        # Basic path configuration
        self.source_dir = config['source_dir']
        self.dataset_output_dir = config['dataset_output_dir']
        self.gt_output_dir = config['gt_output_dir']
        
        # Dataset split ratios
        self.split_ratio = config['split_ratio']
        
        # Additional parameters
        self.num_datasets = config.get('num_datasets', 5)
        self.random_seed = config.get('random_seed', 42)
        
        # Container for file lists of multiple datasets
        self.dataset_files = {i: {'val': [], 'test': []} for i in range(self.num_datasets)}
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create the necessary output directory structure."""
        # Create directories for each dataset and split
        for dataset_idx in range(self.num_datasets):
            for split in ['train', 'val', 'test']:
                for subdir in ['images', 'labels']:
                    os.makedirs(
                        os.path.join(self.dataset_output_dir, f'dataset_{dataset_idx}', split, subdir), 
                        exist_ok=True
                    )
        
        # Create the GT file output directory
        os.makedirs(self.gt_output_dir, exist_ok=True)
        
    def _convert_to_yolo_format(self, points, img_width, img_height):
        """
        Convert bounding box coordinates to YOLO format.

        Args:
            points (list): Bounding box coordinates in the format [[x1, y1], [x2, y2]].
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            list: Normalized YOLO format coordinates [x_center, y_center, width, height].
        """
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Ensure coordinates are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        # Calculate center coordinates and dimensions
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        
        # Normalize coordinates
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]

    def _process_single_json(self, json_path, output_dir):
        """
        Process a single JSON file to generate a corresponding YOLO-format txt file.

        Args:
            json_path (str): Path to the JSON annotation file.
            output_dir (str): Output directory for the YOLO-format txt file.

        Returns:
            str: The base name of the processed file.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        base_name = os.path.splitext(data['imagePath'])[0]
        
        # Create the txt file
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, 'w') as f:
            for shape in data['shapes']:
                # All categories are set to 0
                class_id = 0
                # Convert coordinates to YOLO format
                yolo_coords = self._convert_to_yolo_format(shape['points'], img_width, img_height)
                # Write to txt file
                f.write(f"{class_id} {' '.join(map(str, yolo_coords))}\n")
        
        return base_name

    def prepare_yolo_dataset(self):
        """
        Prepare multiple YOLO-format training datasets using a fixed random seed for reproducibility.

        Returns:
            dict: A dictionary containing the file lists for validation and test sets for each dataset.
        """
        # Get all JSON files from the source directory
        json_files = [f for f in os.listdir(self.source_dir) if f.endswith('.json')]
        fcat6_files = [f for f in json_files if f.startswith('FCAT6')]
        fcat10_files = [f for f in json_files if f.startswith('FCAT10')]
        
        total_files = len(json_files)
        target_train_size = int(total_files * self.split_ratio['train'])
        target_val_size = int(total_files * self.split_ratio['val'])
        target_test_size = int(total_files * self.split_ratio['test'])
        
        # Validate that there are enough files for each split
        if len(fcat10_files) < target_train_size:
            raise ValueError(f"Insufficient number of FCAT10 files ({len(fcat10_files)}) for training set ({target_train_size})")
        if len(fcat6_files) < target_test_size:
            raise ValueError(f"Insufficient number of FCAT6 files ({len(fcat6_files)}) for test set ({target_test_size})")
        
        # Split the files for each dataset
        for dataset_idx in range(self.num_datasets):
            # Set random seed
            random.seed(self.random_seed + dataset_idx)
            
            # Shuffle file lists
            fcat6_files_shuffled = fcat6_files.copy()
            fcat10_files_shuffled = fcat10_files.copy()
            random.shuffle(fcat6_files_shuffled)
            random.shuffle(fcat10_files_shuffled)
            
            # Allocate test set (from FCAT6)
            test_files = fcat6_files_shuffled[:target_test_size]
            remaining_fcat6 = fcat6_files_shuffled[target_test_size:]
            
            # Allocate validation set
            val_from_fcat6 = remaining_fcat6[:target_val_size // 2]
            val_from_fcat10 = fcat10_files_shuffled[:target_val_size // 2]
            val_files = val_from_fcat10 + val_from_fcat6
            
            # Allocate training set
            train_files = fcat10_files_shuffled[target_val_size // 2:target_val_size // 2 + target_train_size]
            
            # Store file lists for the current dataset
            self.dataset_files[dataset_idx]['val'] = val_files
            self.dataset_files[dataset_idx]['test'] = test_files
            
            print(f"\nDataset {dataset_idx} splitting statistics:")
            print(f"Training set: {len(train_files)} files")
            print(f"Validation set: {len(val_files)} files")
            print(f"Test set: {len(test_files)} files")
            
            # Process files for each split
            for split, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
                for json_file in file_list:
                    json_path = os.path.join(self.source_dir, json_file)
                    # Process JSON file to generate txt label file
                    base_name = self._process_single_json(
                        json_path,
                        os.path.join(self.dataset_output_dir, f'dataset_{dataset_idx}', split, 'labels')
                    )
                    
                    # Copy the corresponding image file
                    img_path = os.path.join(self.source_dir, f"{base_name}.png")
                    if os.path.exists(img_path):
                        shutil.copy(
                            img_path,
                            os.path.join(self.dataset_output_dir, f'dataset_{dataset_idx}', split, 'images')
                        )
        
        # Return the file lists of validation and test sets for all datasets
        return self.dataset_files
        
    def generate_gt_files(self):
        """
        Generate ground truth (GT) files for each dataset:
          - Use the validation and test file lists for each dataset to generate corresponding GT files in COCO format.
          - The file naming format is: dataset_{idx}_GT_{split}.json.
        """
        def create_coco_annotation(json_data, image_id, ann_id):
            """
            Convert individual annotation to COCO format.

            Args:
                json_data (dict): Original JSON data.
                image_id (int): ID of the image.
                ann_id (int): Starting annotation ID.

            Returns:
                tuple: (List of COCO annotations, updated annotation ID)
            """
            annotations = []
            for shape in json_data['shapes']:
                points = shape['points']
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                # Ensure coordinates are non-negative and within image bounds
                x1 = max(0, min(x1, json_data['imageWidth']))
                y1 = max(0, min(y1, json_data['imageHeight']))
                x2 = max(0, min(x2, json_data['imageWidth']))
                y2 = max(0, min(y2, json_data['imageHeight']))
                
                # Compute bbox parameters for COCO format
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                # Ensure width and height are greater than 0
                if width <= 0 or height <= 0:
                    continue
                    
                annotation = {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': 1,  # Fixed category
                    'bbox': [float(x_min), float(y_min), float(width), float(height)],
                    'area': float(width * height),
                    'iscrowd': 0,
                    'segmentation': []  # Required by COCO format
                }
                annotations.append(annotation)
                ann_id += 1
                
            return annotations, ann_id

        def generate_gt_json(file_list, output_path):
            """
            Generate a GT JSON file for the specified file list.

            Args:
                file_list (list): List of JSON file names.
                output_path (str): Path to save the generated GT JSON file.
            """
            coco_format = {
                'info': {
                    'description': 'Plant Detection Dataset',
                    'version': '1.0',
                    'year': 2024,
                    'contributor': 'anonymous',
                    'date_created': '2024/12'
                },
                'licenses': [{
                    'id': 1,
                    'name': 'Unknown',
                    'url': 'Unknown'
                }],
                'images': [],
                'annotations': [],
                'categories': [{
                    'id': 1,
                    'name': 'plant',
                    'supercategory': 'none'
                }]
            }
            
            image_id = 1
            ann_id = 1
            
            for file_name in file_list:
                # Read the original JSON file
                json_path = os.path.join(self.source_dir, file_name)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Add image information
                    image_info = {
                        'id': image_id,
                        'file_name': json_data['imagePath'],
                        'width': json_data['imageWidth'],
                        'height': json_data['imageHeight'],
                        'date_captured': '2024-01-01 00:00:00',
                        'license': 1,
                        'coco_url': '',
                        'flickr_url': ''
                    }
                    coco_format['images'].append(image_info)
                    
                    # Add annotation information
                    annotations, ann_id = create_coco_annotation(json_data, image_id, ann_id)
                    coco_format['annotations'].extend(annotations)
                    
                    image_id += 1
                except Exception as e:
                    print(f"Error processing {json_path}: {str(e)}")
                    continue
            
            # Save the GT file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(coco_format, f, indent=2, ensure_ascii=False)

        print("Generating GT files...")
        
        # Generate GT files for each dataset
        for dataset_idx, files in self.dataset_files.items():
            print(f"\nProcessing dataset {dataset_idx}")
            print(f"Number of validation files: {len(files['val'])}")
            print(f"Number of test files: {len(files['test'])}")

            # Generate GT file for validation set
            val_gt_path = os.path.join(self.gt_output_dir, f'dataset_{dataset_idx}_GT_val.json')
            generate_gt_json(files['val'], val_gt_path)
            print(f"Generated GT_val.json for dataset {dataset_idx}")
            
            # Generate GT file for test set
            test_gt_path = os.path.join(self.gt_output_dir, f'dataset_{dataset_idx}_GT_test.json')
            generate_gt_json(files['test'], test_gt_path)
            print(f"Generated GT_test.json for dataset {dataset_idx}")
    
    def run(self):
        """Run the complete pre-training process."""
        self.prepare_yolo_dataset()
        self.generate_gt_files()


# Post-training processing
class PostTrainingProcessor:
    def __init__(self, config):
        """
        Post-training configuration.

        Args:
            config (dict): Configuration dictionary with keys:
                - model_path: Path to the YOLO model weights (e.g., '/path/to/your/model/file').
                - val_data_path: Path to the validation images directory.
                - test_data_path: Path to the test images directory.
                - output_dir: Output directory to save prediction result JSON files.
                - img_size: Input image size.
                - conf_thres: Confidence threshold.
                - iou_thres: IoU threshold for NMS.
        """
        self.model_path = config['model_path']
        self.val_data_path = config['val_data_path']
        self.test_data_path = config['test_data_path']
        self.output_dir = config['output_dir']
        self.img_size = config.get('img_size', 800)
        self.conf_thres = config.get('conf_thres', 0.001)
        self.iou_thres = config.get('iou_thres', 0.001)
        
        # Create the output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def predict_and_save(self, model, data_path, output_json_path):
        """
        Use the model to perform prediction and save the results in COCO format.

        Args:
            model: The YOLO model for inference.
            data_path (str): Path to the directory containing images for prediction.
            output_json_path (str): Path to save the generated JSON file.

        Returns:
            list: List of predictions in COCO format.
        """
        # Load the YOLO model and perform prediction
        results = model.predict(
            source=data_path,
            imgsz=self.img_size,
            conf=self.conf_thres,
            iou=self.iou_thres,
            save=False  # Do not save images
        )
        
        # Convert results to COCO format
        predictions = []
        for i, result in enumerate(results):
            img_path = result.path
            img_name = os.path.basename(img_path)
            boxes = result.boxes
            
            # Retrieve image dimensions
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
                img.close()  # Ensure the image is closed properly
            except Exception as e:
                print(f"Error opening image {img_path}: {str(e)}")
                continue
            
            # Process each detection box
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                # Constrain coordinates within the image
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                # COCO format requires [x, y, width, height]
                width = x2 - x1
                height = y2 - y1
                
                # Skip invalid detection boxes
                if width <= 0 or height <= 0:
                    continue
                
                prediction = {
                    'image_id': i + 1,  # Image ID starting from 1
                    'category_id': 1,    # Fixed category id
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'score': float(conf)
                }
                predictions.append(prediction)
        
        # Save predictions to JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2)
            
        return predictions

    def generate_predictions(self):
        """
        Generate prediction results:
          - Load the trained YOLO model.
          - Perform prediction on both the validation and test datasets.
          - Generate standard val.json and test.json prediction files.
        """
        try:
            # Load the YOLO model
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            
            # Process validation dataset
            val_json_path = os.path.join(self.output_dir, 'val.json')
            self.predict_and_save(model, self.val_data_path, val_json_path)
            print(f"Generated validation predictions: {val_json_path}")
            
            # Process test dataset
            test_json_path = os.path.join(self.output_dir, 'test.json')
            self.predict_and_save(model, self.test_data_path, test_json_path)
            print(f"Generated test predictions: {test_json_path}")
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
    
    def run(self):
        """Run the complete post-training processing."""
        print("Starting post-training processing...")
        self.generate_predictions()
        print("Post-training processing completed.")


def main_pre_training():
    config = {
        'source_dir': '/path/to/your/original_dataset',
        'dataset_output_dir': '/path/to/your/yolo_dataset',
        'gt_output_dir': '/path/to/your/gt_files',
        'split_ratio': {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
    }
    
    processor = PreTrainingProcessor(config)
    processor.run()

def main_post_training():
    config = {
        'model_path': '/path/to/your/model/file',
        'val_data_path': '/path/to/your/val/images',
        'test_data_path': '/path/to/your/test/images',
        'output_dir': '/path/to/your/output_directory',
        'img_size': 800,
        'conf_thres': 0.001,  # Low threshold for later calibration
        'iou_thres': 0.65
    }
    
    processor = PostTrainingProcessor(config)
    processor.run()
