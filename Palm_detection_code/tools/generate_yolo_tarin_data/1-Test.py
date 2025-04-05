"""
This script tests the pre-training processor for generating a YOLO-format dataset along with
ground truth (GT) files. It specifies the dataset configuration, prepares the YOLO dataset,
generates GT files, and verifies their format correctness.

Note:
  - All file paths are represented using generic placeholders to avoid exposing personal information.
  - The original logic of the code is preserved.
"""

import Total
import os
import json

def test_pre_training_processor():
    # Test configuration
    config = {
        'source_dir': '/path/to/your/original_dataset',  # Original dataset path
        'dataset_output_dir': '/path/to/your/yolo_dataset',  # YOLO dataset output path
        'gt_output_dir': '/path/to/your/gt_files',  # GT files output path
        'split_ratio': {'train': 0.8, 'val': 0.1, 'test': 0.1},  # Dataset split ratios
        'num_datasets': 5,  # Generate 5 datasets
        'random_seed': 42  # Random seed
    }
    
    # Initialize the pre-training processor
    processor = Total.PreTrainingProcessor(config)
    
    print(f"\n--------Start generating YOLO dataset--------\n")
    # Prepare the YOLO dataset
    dataset_files = processor.prepare_yolo_dataset()
    
    # Print file statistics for each dataset
    for dataset_idx, files in dataset_files.items():
        print(f"\nDataset {dataset_idx} file statistics:")
        print(f"✓ Number of validation files: {len(files['val'])}")
        print(f"✓ Number of test files: {len(files['test'])}")

    print(f"\n--------Start generating GT files--------\n")
    
    # Generate GT files
    processor.generate_gt_files()
    
    # Verify that GT files have been generated successfully
    for dataset_idx in range(config['num_datasets']):
        val_gt_path = os.path.join(config['gt_output_dir'], f'dataset_{dataset_idx}_GT_val.json')
        test_gt_path = os.path.join(config['gt_output_dir'], f'dataset_{dataset_idx}_GT_test.json')
        assert os.path.exists(val_gt_path), f"dataset_{dataset_idx}_GT_val.json was not generated"
        assert os.path.exists(test_gt_path), f"dataset_{dataset_idx}_GT_test.json was not generated"
        
        # Verify the correctness of the GT file format
        def verify_gt_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check that necessary keys exist
                assert 'images' in data, "GT file missing 'images' key"
                assert 'annotations' in data, "GT file missing 'annotations' key"
                assert 'categories' in data, "GT file missing 'categories' key"
                # Check that the data is not empty
                assert len(data['images']) > 0, "GT file contains no image information"
                print(f"✓ {os.path.basename(file_path)} format verified successfully")
                print(f"  - Number of images: {len(data['images'])}")
                print(f"  - Number of annotations: {len(data['annotations'])}")
        
        verify_gt_file(val_gt_path)
        verify_gt_file(test_gt_path)
    
    print("\n✓ All tests passed")


if __name__ == "__main__":
    test_pre_training_processor()
