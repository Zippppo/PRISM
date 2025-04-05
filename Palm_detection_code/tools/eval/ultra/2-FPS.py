"""
This script evaluates the performance (FPS) of an object detection model using both single
image and batch processing modes. It measures inference speed, analyzes GPU memory and 
utilization metrics, and outputs detailed performance data for various batch sizes.

Notes:
- Replace any file paths with generic ones (e.g., '/path/to/your/model/file', '/path/to/your/image/file')
  to avoid including personal information.
- The original logic of the code has been preserved.
"""

from ultralytics import YOLO, RTDETR
import torch
import time
import numpy as np
import cv2
import warnings
from torch.cuda import amp
from contextlib import contextmanager

# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@contextmanager
def suppress_ultralytics_warnings():
    """Temporarily suppress warnings from ultralytics."""
    import logging
    logger = logging.getLogger("ultralytics")
    current_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(current_level)

class YOLOFPSTester:
    def __init__(self, model_path, img_size=800):
        """
        Initialize the FPS tester with the given model and image size.
        Args:
            model_path (str): Path to the model file (e.g., '/path/to/your/model/file').
            img_size (int): Size of the input image (width and height).
        """
        self.device = torch.device('cuda')
        # Uncomment the following line if you wish to use the YOLO model instead
        # self.model = YOLO(model_path)
        self.model = RTDETR(model_path)
        self.img_size = img_size
        self.model.model.eval()
        self.model.model.to(self.device)
    
    def prepare_dummy_batch(self, batch_size):
        """
        Generate a dummy batch of normalized random data in the range [0, 1].
        Args:
            batch_size (int): Number of images in the batch.
        Returns:
            torch.Tensor: Batch of dummy images.
        """
        return torch.rand(batch_size, 3, self.img_size, self.img_size).to(self.device)
    
    def prepare_image(self, image_path):
        """
        Load and process an image for inference.
        Args:
            image_path (str): Path to the image file (e.g., '/path/to/your/image/file').
        Returns:
            torch.Tensor: Processed image tensor.
        """
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device)
        img /= 255.0  # Normalize to the range [0, 1]
        return img
    
    def test_single_image_fps(self, img, num_warmup=50, num_test=100):
        """
        Test the FPS for a single image.
        Args:
            img (torch.Tensor): An image tensor.
            num_warmup (int): Number of warm-up iterations.
            num_test (int): Number of test iterations.
        Returns:
            tuple: Mean FPS and standard deviation of FPS.
        """
        print(f"\nWarming up for {num_warmup} iterations...")
        
        with suppress_ultralytics_warnings():
            # Warm-up phase
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
                for _ in range(num_warmup):
                    _ = self.model.predict(img, verbose=False)
            
            torch.cuda.synchronize()
            
            # Testing phase
            print(f"Starting single image test for {num_test} iterations...")
            times = []
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
                for i in range(num_test):
                    start_time = time.perf_counter()
                    _ = self.model.predict(img, verbose=False)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start_time
                    times.append(elapsed)
                    if (i + 1) % 20 == 0:
                        print(f"Completed {i + 1}/{num_test} iterations")
        
        # Compute statistics
        times = np.array(times)
        mean_fps = 1.0 / np.mean(times)
        std_fps = np.std(1.0 / times)
        
        return mean_fps, std_fps
    
    def analyze_gpu_metrics(self, batch_size):
        """
        Analyze GPU metrics.
        Args:
            batch_size (int): Batch size used during inference.
        Returns:
            dict or None: Dictionary containing GPU metrics or None if an error occurs.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming the first GPU is used
            
            # Retrieve GPU information
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            total_memory = memory_info.total / 1024**2  # MB
            used_memory = memory_info.used / 1024**2    # MB
            free_memory = memory_info.free / 1024**2    # MB
            gpu_util = utilization.gpu                  # GPU utilization percentage
            memory_util = utilization.memory            # Memory utilization percentage
            
            print(f"\nGPU Analysis (Batch Size = {batch_size}):")
            print(f"{'='*50}")
            print(f"Total Memory: {total_memory:.2f} MB")
            print(f"Used Memory: {used_memory:.2f} MB ({used_memory/total_memory*100:.1f}%)")
            print(f"Free Memory: {free_memory:.2f} MB ({free_memory/total_memory*100:.1f}%)")
            print(f"GPU Utilization: {gpu_util}%")
            print(f"Memory Utilization: {memory_util}%")
            
            return {
                'total_memory': total_memory,
                'used_memory': used_memory,
                'free_memory': free_memory,
                'gpu_util': gpu_util,
                'memory_util': memory_util
            }
        except ImportError:
            print("Please install pynvml: pip install pynvml")
            return None
        except Exception as e:
            print(f"Error obtaining GPU metrics: {str(e)}")
            return None

    def test_batch_fps(self, batch_size=32, num_warmup=50, num_test=100):
        """
        Test the FPS for batch processing.
        Args:
            batch_size (int): Number of images per batch.
            num_warmup (int): Number of warm-up iterations.
            num_test (int): Number of test iterations.
        Returns:
            tuple: Mean FPS and standard deviation of FPS.
        """
        print(f"\nPreparing batch processing data (batch_size={batch_size})...")
        batch = self.prepare_dummy_batch(batch_size)
        
        # Monitor GPU memory usage
        initial_memory = torch.cuda.memory_allocated()
        gpu_metrics_before = self.analyze_gpu_metrics(batch_size)
        
        with suppress_ultralytics_warnings():
            # Warm-up phase
            print(f"Warming up for {num_warmup} iterations...")
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
                for _ in range(num_warmup):
                    _ = self.model.predict(batch, verbose=False)
            
            # Record GPU metrics during testing
            gpu_metrics_during = self.analyze_gpu_metrics(batch_size)
            
            torch.cuda.synchronize()
            
            # Testing phase
            print(f"Starting batch test for {num_test} iterations...")
            times = []
            memory_usage = []
            
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
                for i in range(num_test):
                    start_time = time.perf_counter()
                    _ = self.model.predict(batch, verbose=False)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start_time
                    times.append(elapsed)
                    
                    memory_usage.append(torch.cuda.memory_allocated())
                    
                    if (i + 1) % 20 == 0:
                        print(f"Completed {i + 1}/{num_test} iterations")
        
        # Compute statistics
        times = np.array(times)
        mean_fps = batch_size / np.mean(times)
        std_fps = batch_size * np.std(1.0 / times)
        
        # Print detailed performance analysis
        print(f"\nPerformance Analysis for Batch Size {batch_size}:")
        print(f"{'='*50}")
        print(f"Average FPS: {mean_fps:.2f} ± {std_fps:.2f}")
        print(f"Processing time per batch: {np.mean(times)*1000:.2f} ms ± {np.std(times)*1000:.2f} ms")
        print(f"Processing time per image: {np.mean(times)*1000/batch_size:.2f} ms")
        
        # Analyze memory usage
        if gpu_metrics_before and gpu_metrics_during:
            memory_increase = gpu_metrics_during['used_memory'] - gpu_metrics_before['used_memory']
            print(f"\nMemory Usage Analysis:")
            print(f"Memory increase per batch: {memory_increase:.2f} MB")
            print(f"Memory cost per image: {memory_increase/batch_size:.2f} MB")
        
        return mean_fps, std_fps

def main():
    # Replace with generic file paths to avoid personal information
    model_path = '/path/to/your/model/file'
    image_path = '/path/to/your/image/file'  # Optional
    batch_sizes = [1, 4, 6, 7, 8, 9, 10, 12, 16, 32]  # More dense sampling around 8
    
    tester = YOLOFPSTester(model_path, img_size=800)
    results = {}
    
    # Single image performance test
    print("\n" + "="*50)
    print("Single Image Performance Test")
    print("="*50)
    
    # You can choose to use an actual image or dummy data
    test_img = tester.prepare_dummy_batch(1)
    # Alternatively, you can use an actual image:
    # test_img = tester.prepare_image(image_path)
    
    single_mean_fps, single_std_fps = tester.test_single_image_fps(test_img)
    results['single'] = {'mean_fps': single_mean_fps, 'std_fps': single_std_fps}
    
    # Batch performance test
    print("\n" + "="*50)
    print("Batch Performance Test")
    print("="*50)
    
    # Collect performance curve data
    performance_data = {
        'batch_sizes': [],
        'fps': [],
        'memory_usage': [],
        'gpu_util': []
    }
    
    for bs in batch_sizes:
        try:
            print(f"\nTesting batch_size = {bs}")
            mean_fps, std_fps = tester.test_batch_fps(batch_size=bs)
            gpu_metrics = tester.analyze_gpu_metrics(bs)
            
            # Collect data
            performance_data['batch_sizes'].append(bs)
            performance_data['fps'].append(mean_fps)
            if gpu_metrics:
                performance_data['memory_usage'].append(gpu_metrics['used_memory'])
                performance_data['gpu_util'].append(gpu_metrics['gpu_util'])
            
            results[f'batch_{bs}'] = {'mean_fps': mean_fps, 'std_fps': std_fps}
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Batch size {bs} caused a memory overflow")
            torch.cuda.empty_cache()
            break
    
    # Print performance curve analysis
    print("\nPerformance Curve Analysis:")
    print("="*90)
    print(f"{'Batch Size':<12}{'FPS':>12}{'Std Dev':>12}{'Memory Usage (MB)':>20}{'GPU Utilization (%)':>20}")
    print("-"*90)
    for i in range(len(performance_data['batch_sizes'])):
        bs = performance_data['batch_sizes'][i]
        fps = performance_data['fps'][i]
        std = results[f'batch_{bs}']['std_fps']  # Standard deviation
        mem = performance_data['memory_usage'][i] if performance_data['memory_usage'] else 0
        util = performance_data['gpu_util'][i] if performance_data['gpu_util'] else 0
        print(f"{bs:<12}{fps:>12.2f}{std:>12.2f}{mem:>20.2f}{util:>20.1f}")
    
    # Include single image result
    print(f"{'Single Image':<12}{results['single']['mean_fps']:>12.2f}"
          f"{results['single']['std_fps']:>12.2f}{'-':>20}{'-':>20}")

if __name__ == '__main__':
    main()