"""
This script splits a COCO-format dataset into several subsets for training, validation, and testing.
It reads JSON annotation files from a specified source directory, partitions them based on the provided
split ratios, and copies the corresponding images into a standard COCO directory structure. Each split
is saved in its respective directory along with a COCO-format JSON file containing image and annotation information.

Note:
  - Replace file paths with generic placeholders to avoid including personal information.
  - The original logic of the code is preserved.
"""

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

class CocoDatasetSplitter:
    def __init__(self, config):
        """
        Initialize the dataset splitter.

        Args:
            config (dict): Configuration dictionary containing:
                - source_dir: Directory containing the original COCO-format JSON files.
                - output_base_dir: Directory where the output splits will be saved.
                - split_ratio: Dictionary specifying the split ratios for train, val, and test sets.
                - num_datasets: Number of datasets to create (default is 5).
                - random_seed: Random seed (default is 42).
        """
        self.source_dir = Path(config['source_dir'])
        self.output_base_dir = Path(config['output_base_dir'])
        self.split_ratio = config['split_ratio']
        self.num_datasets = config.get('num_datasets', 5)
        self.random_seed = config.get('random_seed', 42)
        
        # Store file lists for multiple datasets
        self.dataset_files = {i: {'val': [], 'test': []} for i in range(self.num_datasets)}
        
        # Create a basic COCO format template
        self.coco_template = {
            "info": {
                "description": "palm detection dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "anonymous",
                "date_created": "YYYY-MM-DD"
            },
            "licenses": [{
                "id": 1,
                "name": "Unknown",
                "url": "Unknown"
            }],
            "images": [],
            "annotations": [],
            "categories": [{
                "id": 1,
                "name": "palm",
                "supercategory": "none"
            }]
        }
        
    def _create_directory_structure(self):
        """Create the standard COCO directory structure."""
        for dataset_idx in range(self.num_datasets):
            dataset_dir = self.output_base_dir / f'dataset_{dataset_idx}'
            
            # Create the 'annotations' directory
            (dataset_dir / 'annotations').mkdir(parents=True, exist_ok=True)
            
            # Create train/val/test directories
            for split in ['train', 'val', 'test']:
                (dataset_dir / split).mkdir(parents=True, exist_ok=True)

    def _load_all_annotations(self):
        """Load all COCO-format annotation JSON files."""
        annotations = []
        for json_file in self.source_dir.glob('*.json'):
            if json_file.name.startswith('FCAT'):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Add file name information for later processing
                    data['_file_name'] = json_file.name
                    annotations.append(data)
        return annotations

    def split_datasets(self):
        """Split the dataset and save in COCO format."""
        self._create_directory_structure()
        
        # Load all annotation files
        all_annotations = self._load_all_annotations()
        
        # Separate FCAT6 and FCAT10 files
        fcat6_files = [a for a in all_annotations if a['_file_name'].startswith('FCAT6')]
        fcat10_files = [a for a in all_annotations if a['_file_name'].startswith('FCAT10')]
        
        total_files = len(all_annotations)
        target_train_size = int(total_files * self.split_ratio['train'])
        target_val_size = int(total_files * self.split_ratio['val'])
        target_test_size = int(total_files * self.split_ratio['test'])
        
        # Validate file counts
        if len(fcat10_files) < target_train_size:
            raise ValueError(f"The number of FCAT10 files ({len(fcat10_files)}) is insufficient for the training set ({target_train_size})")
        if len(fcat6_files) < target_test_size:
            raise ValueError(f"The number of FCAT6 files ({len(fcat6_files)}) is insufficient for the test set ({target_test_size})")
        
        # Split each dataset
        for dataset_idx in range(self.num_datasets):
            print(f"\nProcessing dataset {dataset_idx}")
            
            # Set random seed
            random.seed(self.random_seed + dataset_idx)
            
            # Shuffle file lists
            fcat6_files_shuffled = fcat6_files.copy()
            fcat10_files_shuffled = fcat10_files.copy()
            random.shuffle(fcat6_files_shuffled)
            random.shuffle(fcat10_files_shuffled)
            
            # Assign test set (from FCAT6)
            test_files = fcat6_files_shuffled[:target_test_size]
            remaining_fcat6 = fcat6_files_shuffled[target_test_size:]
            
            # Assign validation set
            val_from_fcat6 = remaining_fcat6[:target_val_size//2]
            val_from_fcat10 = fcat10_files_shuffled[:target_val_size//2]
            val_files = val_from_fcat10 + val_from_fcat6
            
            # Assign training set
            train_files = fcat10_files_shuffled[target_val_size//2:target_val_size//2 + target_train_size]
            
            # Store file lists for the current dataset
            self.dataset_files[dataset_idx]['val'] = val_files
            self.dataset_files[dataset_idx]['test'] = test_files
            
            # Create COCO-format data for each split
            for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
                self._create_coco_split(dataset_idx, split, files)
                
            print(f"Dataset {dataset_idx} split completed:")
            print(f"Training set: {len(train_files)} files")
            print(f"Validation set: {len(val_files)} files")
            print(f"Test set: {len(test_files)} files")

    def _create_coco_split(self, dataset_idx, split, files):
        """Create COCO-format dataset for a specific split."""
        dataset_dir = self.output_base_dir / f'dataset_{dataset_idx}'
        coco_data = self.coco_template.copy()
        
        image_id = 1
        annotation_id = 1
        
        # Initialize new 'images' and 'annotations' lists
        coco_data['images'] = []
        coco_data['annotations'] = []
        
        for file_data in files:
            # Get image file name and path
            image_name = file_data['_file_name'].replace('.json', '.png')
            src_image = self.source_dir / image_name
            dst_image = dataset_dir / split / image_name
            
            # Copy image if it exists
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
                print(f"Copied {image_name} to {split} set in dataset_{dataset_idx}")
            else:
                print(f"Warning: Image {image_name} not found in {self.source_dir}")
            
            # Add image information from the current file only
            for image in file_data['images']:
                image_info = {
                    'id': image_id,
                    'file_name': image_name,
                    'height': image['height'],
                    'width': image['width']
                }
                coco_data['images'].append(image_info)
            
            # Add annotation information for the current image only
            for ann in file_data['annotations']:
                if ann['image_id'] == file_data['images'][0]['id']:  # Ensure the annotation belongs to the current image
                    ann_info = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': ann['category_id'],
                        'bbox': ann['bbox'],
                        'area': ann['area'],
                        'iscrowd': ann['iscrowd']
                    }
                    coco_data['annotations'].append(ann_info)
                    annotation_id += 1
            
            image_id += 1
        
        # Save the COCO-format annotation file
        annotation_file = dataset_dir / 'annotations' / f'{split}.json'
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

def main():
    config = {
        'source_dir': '/path/to/your/coco_format_directory',
        'output_base_dir': '/path/to/your/COCO_5_splits_directory',
        'split_ratio': {
            'train': 0.8,
            'val': 0.1,
            'test': 0.1
        },
        'num_datasets': 5,
        'random_seed': 42
    }
    
    splitter = CocoDatasetSplitter(config)
    splitter.split_datasets()

if __name__ == '__main__':
    main()
