"""
This script trains an object detection model using RTDETR (or optionally YOLO) from the ultralytics library.
It configures necessary settings, initializes the model with the specified weights, and then trains it on
a given dataset. Training parameters such as epochs, image size, and batch size are provided. The script
also demonstrates how to switch between RTDETR and YOLO models if needed.

Note:
  - All sensitive file paths have been replaced with generic placeholders (e.g., '/path/to/your/model/file', '/path/to/your/dataset').
  - The original logic of the code remains unchanged.
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import numpy as np
import ultralytics
from ultralytics import YOLO, RTDETR
from ultralytics.utils import DEFAULT_CFG
import cv2
import time
import torch
from collections import deque

# Run ultralytics checks to ensure the environment is properly set up
ultralytics.checks()

# Dataset index and model name configuration
i = 0
name = "RTDETR"

# Initialize the model using RTDETR. Replace the model path with your own model file path.
model = RTDETR('/path/to/your/model/file/rtdetr-x.pt')

# Uncomment the following line to use the YOLO model instead
# model = YOLO('/path/to/your/model/file/yolov10x.pt')

# Train the model using the specified dataset configuration and training parameters.
results = model.train(
    data=f"/path/to/your/dataset/dataset_{i}/data.yaml",
    epochs=100,
    imgsz=800,
    single_cls=True,
    device='device=0',  # Specify the device, e.g., 'cuda:0'
    batch=8,
    amp=False,  # amp=False can help to prevent output values from becoming NaN
    project=f"/path/to/your/output_directory/{name}/dataset_{i}",
    name='100epochs'
)

# Uncomment the following lines for an alternative YOLO model configuration with different parameters:
# model = YOLO('/path/to/your/model/file/yolov8x.pt')
# results = model.train(
#     data=f"/path/to/your/dataset/dataset_{i}/data.yaml",
#     epochs=200,
#     imgsz=800,
#     single_cls=True,
#     device='device=0',
#     batch=8,
#     amp=False,  # amp=False can help to prevent output values from becoming NaN
#     project=f"/path/to/your/output_directory/{name}/dataset_{i}",
#     name='200epochs'
# )
