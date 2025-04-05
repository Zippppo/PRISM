"""
Detection Model Calibration Script

This script performs calibration on various object detection models using different calibration methods.
It evaluates and visualizes the calibration results before and after applying different calibration techniques.

Key features:
- Supports multiple detection models and calibration methods
- Generates calibration plots for analysis
- Handles multiple confidence thresholds
"""

import os
import sys
sys.path.append('/path/to/detection_calibration/src')  # Update with your project path

from detection_calibration.DetectionCalibration import DetectionCalibration
from detection_calibration.utils import threshold_detections, load_detections_from_file

# Define model names for calibration
model_names = ["RTDETR", "YOLO_V8_X", "YOLO_V9_E", "YOLO_V10_X", "YOLO_V11_X"]

# Define confidence thresholds
thresholds = [0.1, 0.001, 0.05, 0.01]

# Define calibrator types and their display names
calibrator_types = ['isotonic_regression', 'platt_scaling', 
                   'temperature_scaling', 'linear_regression']

method_names = {
    'isotonic_regression': 'Isotonic Regression',
    'platt_scaling': 'Platt Scaling',
    'temperature_scaling': 'Temperature Scaling',
    'linear_regression': 'Linear Regression'
}

def create_directory(path):
    """Create directory if it doesn't exist."""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created successfully or already exists.")
    except Exception as e:
        print(f"Error creating directory: {e}")

for threshold in thresholds:
    for model_name in model_names:
        print(f"\n\n=================== Processing model: {model_name}, threshold: {threshold} ===================")
        
        # Define input file paths
        test_json = f"/path/to/ultra/ultra_json_files_thr_{threshold}/{model_name}/test.json"
        val_json = f"/path/to/ultra/ultra_json_files_thr_{threshold}/{model_name}/val.json"
        gt_test_json = '/path/to/ultra/dataset_4_GT_test.json'
        gt_val_json = '/path/to/ultra/dataset_4_GT_val.json'

        # Create output directory
        output_dir = f'/path/to/output/cali_res/{threshold}/{model_name}'
        create_directory(output_dir)

        # Initialize calibration model
        calibration_model = DetectionCalibration(gt_val_json,
                                               gt_test_json,
                                               bin_count=20)

        print(f"Calibration model bin_count: {calibration_model.calibration_scheme.bin_count}")

        # Generate pre-calibration results
        print("\n----------------------PRE-CALIBRATION RESULTS----------------------")
        calibration_model.evaluate_calibration(
            test_json, 
            plot_path=f'{output_dir}/Before Threshold A.png',
            method_name='Before Threshold A'
        )

        print("\n----------------------POST-CALIBRATION RESULTS----------------------")

        # Get first calibrator's threshold for threshold_A visualization
        first_calibrator, first_thresholds = calibration_model.fit(
            val_json, 
            calibrator_type=calibrator_types[0]
        )

        # Visualize results after threshold_A filtering
        test_detections = load_detections_from_file(test_json)
        thr_A_test_detect = threshold_detections(
            test_detections,
            first_thresholds[0],  # threshold_A
            calibration_model.calibration_scheme.dataset_classes
        )

        calibration_model.evaluate_calibration(
            thr_A_test_detect,
            plot_path=f'{output_dir}/After Threshold A.png',
            method_name='After Threshold A'
        )

        # Generate calibration results for each method
        for cal_type in calibrator_types:
            print(f"\n\nTesting calibrator: {cal_type}")
            print("="*50)
            
            try:
                if cal_type == calibrator_types[0]:
                    # Use previously trained first calibrator
                    calibrator, thresholds_cal = first_calibrator, first_thresholds
                else:
                    # Train new calibrator for other methods
                    calibrator, thresholds_cal = calibration_model.fit(
                        val_json, 
                        calibrator_type=cal_type
                    )
                
                # Generate results after complete calibration process
                cal_test_detections = calibration_model.transform(
                    test_json,
                    calibrator, 
                    thresholds_cal
                )
                
                calibration_model.evaluate_calibration(
                    cal_test_detections,
                    plot_path=f'{output_dir}/{cal_type}.png',
                    method_name=method_names[cal_type]
                )
            except Exception as e:
                print(f"Error with {cal_type}: {str(e)}")
                continue