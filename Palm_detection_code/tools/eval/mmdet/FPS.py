"""
Description:
This script measures the frames per second (FPS) performance of object detection models.
It provides functions to measure the FPS for single image inference and for batch processing.
For single image inference, the model is warmed up for a number of iterations before timing a specified number
of iterations. For batch processing, a preprocessed input tensor is used to create a batch input,
and similar timing is performed.
The script supports different model types (e.g., DeformableDETR, DINO) and outputs the mean and standard deviation of the FPS.
All file paths and directories use generic placeholders to avoid exposing personal information.

Usage:
    python FPS.py --data_root /path/to/your/dataset --configs_folder /path/to/your/configs \
    --checkpoint_folder /path/to/your/checkpoints --device cuda:0
"""

import os
import torch
import time
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector
import numpy as np
import json
from mmdet.models import DeformableDETR, DINO

def get_latest_checkpoint(method):
    """
    Get the latest checkpoint file for the specified method.

    Parameters:
        method (str): The method name.

    Returns:
        str: The path read from the checkpoint file.
    """
    checkpoint_file = f'/path/to/your/work_dirs/{method}/last_checkpoint'
    with open(checkpoint_file, 'r') as f:
        return f.read().strip()

def get_first_image_path(ann_file, data_root):
    """
    Retrieve the path of the first image from a COCO-format annotation file.

    Parameters:
        ann_file (str): Path to the COCO annotation JSON file.
        data_root (str): Root directory of the dataset.

    Returns:
        str: Full path to the first image file.
    """
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Get the filename of the first image
    first_image = annotations['images'][0]
    image_path = os.path.join(data_root, 'val', first_image['file_name'])
    return image_path

def measure_fps(model, img, warmup=10, num_samples=100):
    """
    Measure the FPS for single image inference.

    Parameters:
        model: The detection model.
        img (str): Path to the input image.
        warmup (int): Number of warmup iterations.
        num_samples (int): Number of iterations for timing.

    Returns:
        tuple: Mean and standard deviation of FPS.
    """
    # Warm up the model
    print(f"\nWarming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = inference_detector(model, img)
    
    # Timing
    print(f"Timing inference for ({num_samples} iterations)...")
    times = []
    with torch.no_grad():
        for i in range(num_samples):
            start_time = time.perf_counter()
            _ = inference_detector(model, img)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_samples} iterations")
    
    # Calculate metrics
    times = np.array(times)
    mean_fps = 1.0 / np.mean(times)
    std_fps = np.std(1.0 / times)
    
    return mean_fps, std_fps

def measure_fps_batch(model, img, batch_size=32, warmup=10, num_samples=100):
    """
    Measure the FPS for batch processing inference.

    Parameters:
        model: The detection model.
        img (str): Path to the input image.
        batch_size (int): Batch size for processing.
        warmup (int): Number of warmup iterations.
        num_samples (int): Number of iterations for timing.

    Returns:
        tuple: Mean and standard deviation of FPS for batch processing.
    """
    print("\nPreparing batch processing data...")
    with torch.no_grad():
        # Build the test pipeline
        print("Building test pipeline...")
        cfg = model.cfg
        test_pipeline = cfg.test_dataloader.dataset.pipeline
        from mmengine.dataset import Compose
        pipeline = Compose(test_pipeline)
        
        # Process the input image
        print("Processing input image...")
        data = dict(img_path=img, img_id=0)
        data = pipeline(data)
        
        # Extract the input tensor
        print("Extracting input tensor...")
        input_tensor = data['inputs']
        print(f"Input tensor dtype: {input_tensor.dtype}")
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Ensure the tensor is float32
        if input_tensor.dtype != torch.float32:
            print("Converting tensor dtype to float32...")
            input_tensor = input_tensor.float()
            print(f"Converted tensor dtype: {input_tensor.dtype}")
        
        # Create batch input by repeating the tensor
        print(f"Creating batch input (batch_size={batch_size})...")
        batch_inputs = input_tensor.repeat(batch_size, 1, 1, 1)
        print(f"Batch input shape: {batch_inputs.shape}")
        print(f"Batch input dtype: {batch_inputs.dtype}")
        
        # Move data to GPU
        batch_inputs = batch_inputs.cuda()
        
        # Warm up for batch processing
        print(f"\nWarming up ({warmup} iterations) for batch processing...")
        for _ in range(warmup):
            if isinstance(model, (DeformableDETR, DINO)):
                if _ == 0:
                    print("Using DETR inference method")
                _ = model.extract_feat(batch_inputs)
            else:
                if _ == 0:
                    print("Using RTMDet inference method")
                feat = model.backbone(batch_inputs)
                _ = model.neck(feat)
    
    # Timing for batch processing
    print(f"\nTiming batch processing for ({num_samples} iterations)...")
    times = []
    with torch.no_grad():
        for i in range(num_samples):
            start_time = time.perf_counter()
            if isinstance(model, (DeformableDETR, DINO)):
                _ = model.extract_feat(batch_inputs)
            else:
                feat = model.backbone(batch_inputs)
                _ = model.neck(feat)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_samples} iterations")
    
    # Calculate metrics for batch processing
    times = np.array(times)
    mean_fps = batch_size / np.mean(times)
    std_fps = batch_size * np.std(1.0 / times)
    
    return mean_fps, std_fps

def main(args):
    """
    Main function to load model configurations, initialize models, and measure FPS performance.
    
    Parameters:
        args: Parsed command-line arguments.
    
    The function processes each method defined in the methods_and_checkpoints dictionary,
    measures single image and batch processing FPS, and prints a performance summary.
    """
    # Define methods and corresponding checkpoint files using placeholder paths
    methods_and_checkpoints = {
        'ddq': os.path.join(args.checkpoint_folder, 'ddq_best.pth'),
        'dino': os.path.join(args.checkpoint_folder, 'dino_best.pth')
    }
    
    # Define dataset paths
    data_root = args.data_root
    ann_file = os.path.join(data_root, 'annotations', 'val.json')
    
    # Get the test image path from the annotation file
    test_img = get_first_image_path(ann_file, data_root)
    print(f"Using test image: {test_img}")
    
    results = {}
    
    for method, checkpoint_file in methods_and_checkpoints.items():
        print("\n" + "="*20 + f" Testing {method.upper()} " + "="*20)
        
        # Define configuration file path using placeholder directory
        config_file = os.path.join(args.configs_folder, method, f"{method}_4.py")
        
        # Initialize model configuration
        cfg = Config.fromfile(config_file)
        cfg.test_dataloader.dataset.ann_file = ann_file
        cfg.test_dataloader.dataset.data_root = data_root
        
        # Remove custom imports if present
        if 'custom_imports' in cfg:
            cfg.pop('custom_imports')
            
        init_default_scope(cfg.get('default_scope', 'mmdet'))
        model = init_detector(config_file, checkpoint_file, device=args.device)
        
        # Measure FPS for single image inference
        print("\nTesting single image FPS:")
        mean_fps, std_fps = measure_fps(model, test_img, warmup=args.single_warmup, num_samples=args.single_num_samples)
        results[method] = {'single_mean_fps': mean_fps, 'single_std_fps': std_fps}
        
        # Measure FPS for batch processing inference
        print("\nTesting batch processing FPS:")
        batch_mean_fps, batch_std_fps = measure_fps_batch(model, test_img, batch_size=args.batch_size,
                                                          warmup=args.batch_warmup, num_samples=args.batch_num_samples)
        results[method].update({'batch_mean_fps': batch_mean_fps, 'batch_std_fps': batch_std_fps})
        
        print(f"\n{method.upper()} results:")
        print(f"Single Image FPS: {mean_fps:.2f} ± {std_fps:.2f}")
        print(f"Batch Processing FPS: {batch_mean_fps:.2f} ± {batch_std_fps:.2f}")
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Print performance summary
    print("\n" + "="*70)
    print("Performance Summary:")
    print("-" * 70)
    print(f"{'Method':<12}{'Single FPS':>15}{'Single Std':>15}{'Batch FPS':>15}{'Batch Std':>15}")
    print("-" * 70)
    
    for method in methods_and_checkpoints.keys():
        single_fps = results[method]['single_mean_fps']
        single_std = results[method]['single_std_fps']
        batch_fps = results[method]['batch_mean_fps']
        batch_std = results[method]['batch_std_fps']
        print(f"{method:<12}{single_fps:>15.2f}{single_std:>15.2f}{batch_fps:>15.2f}{batch_std:>15.2f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure FPS performance for object detection inference."
    )
    parser.add_argument(
        "--data_root", type=str, default="/path/to/your/dataset",
        help="Path to the dataset root directory (default: /path/to/your/dataset)"
    )
    parser.add_argument(
        "--configs_folder", type=str, default="/path/to/your/configs",
        help="Path to the configuration files folder (default: /path/to/your/configs)"
    )
    parser.add_argument(
        "--checkpoint_folder", type=str, default="/path/to/your/checkpoints",
        help="Path to the folder containing model checkpoints (default: /path/to/your/checkpoints)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run inference on (default: cuda:0)"
    )
    parser.add_argument(
        "--single_warmup", type=int, default=10,
        help="Number of warmup iterations for single image inference (default: 10)"
    )
    parser.add_argument(
        "--single_num_samples", type=int, default=100,
        help="Number of iterations for timing single image inference (default: 100)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for batch processing FPS measurement (default: 16)"
    )
    parser.add_argument(
        "--batch_warmup", type=int, default=10,
        help="Number of warmup iterations for batch processing (default: 10)"
    )
    parser.add_argument(
        "--batch_num_samples", type=int, default=100,
        help="Number of iterations for timing batch processing (default: 100)"
    )
    
    args = parser.parse_args()
    main(args) 

#    export PYTHONPATH=/data/general/development/kangning/DETECTION/mmdetection:$PYTHONPATH