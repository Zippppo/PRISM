# PRISM
> code for "Detection and Geographic Localization of Natural Objects in the Wild: A Case Study on Palms"

## 📰 News
* **[2025-04-28]** 🎉 Our paper, "Detection and Geographic Localization of Natural Objects in the Wild: A Case Study on Palms," has been accepted by IJCAI 2025! We are thrilled to share our work with the community. 🥳

## Overview
This repository contains a comprehensive system for palm tree detection, segmentation, and analysis using aerial/satellite imagery. The codebase provides tools for training object detection models, processing orthomosaic images, and analyzing detection results for palm tree counting and monitoring.

## 🌴 Dataset (DataPrepare)

* **Full Dataset Download:** [PALMS](https://drive.google.com/file/d/1z-DnZUN4LOOOk6TrPZ6JuQFhSFb9WKpL/view?usp=drive_link)


## Repository Structure

```
Palm_detection_code/     # Main code directory
├── Train/               # Model training scripts
│   ├── train_ultra.py   # Training using Ultralytics framework (YOLO, RTDETR)
│   └── train_mmdet.py   # Training using MMDetection framework (DINO)
├── SAM/                 # Segment Anything Model implementation
│   ├── Orthomosaic/     # Scripts for processing large orthomosaic images
│   │   ├── 1-predict.py # Detection on orthomosaic using sliding window
│   │   ├── 2-nms.py     # Non-maximum suppression for overlapping detections
│   │   └── 3-segment.py # Segmentation of detected palms
│   └── Raw_data/        # Processing for raw imagery
├── Calibration/         # Model calibration tools
│   ├── calibration_models/ # Temperature scaling and other calibration methods
│   └── generate_json_files/ # Prepare data for calibration
├── tools/               # Utility tools for the project
│   ├── counting/        # Palm counting tools from detection results
│   ├── saliency_map/    # Visualization of model attention
│   ├── eval/            # Evaluation scripts for model performance
│   ├── generate_yolo_tarin_data/ # Data preparation for YOLO/RTDETR
│   └── generate_mmdet_train_data/ # Data preparation for MMDetection models
└── requirements.txt     # Python dependencies

Data_example/            # Example dataset structure
├── images/              # Example training images
├── labels/              # Corresponding annotations in YOLO format
└── ReadMe.txt           # Note about data availability
```

## Requirements
The project depends on the following main packages:

```
torch==2.0.1
torchvision==0.15.2
ultralytics==8.3.38
mmcv==2.2.0
mmengine==0.10.6
opencv_python==4.9.0.80
rasterio==1.4.3
Pillow==11.1.0
numpy==1.24.1
pandas==2.2.3
Shapely==2.0.7
PyYAML==6.0.2
tqdm==4.66.4
```

## Installation

```bash
git clone <repository-url>
cd Palm_detection_code
pip install -r requirements.txt
```

## Usage

### Training Models

#### Training with Ultralytics (YOLO/RTDETR)
```bash
cd Palm_detection_code/Train
python train_ultra.py
```

#### Training with MMDetection (DINO)
```bash
cd Palm_detection_code/Train
python train_mmdet.py
```

### Processing Orthomosaic Images

The system uses a sliding window approach to process large orthomosaic images and detect palm trees:

1. Run detection on image tiles:
```bash
cd Palm_detection_code/SAM/Orthomosaic
python 1-predict.py
```

2. Apply Non-Maximum Suppression (NMS) to remove duplicate detections:
```bash
python 2-nms.py
```

3. Segment detected palm trees:
```bash
python 3-segment.py
```

### Data Preparation

The repository provides tools for generating training data in both YOLO/RTDETR and MMDetection formats:

```bash
cd Palm_detection_code/tools/generate_yolo_tarin_data
# or
cd Palm_detection_code/tools/generate_mmdet_train_data
```

### Evaluation

Evaluate model performance using the evaluation scripts:

```bash
cd Palm_detection_code/tools/eval/ultra
# or
cd Palm_detection_code/tools/eval/mmdet
```

## Dataset

The repository includes a small example dataset in the `Data_example` directory. The full dataset will be made available upon publication acceptance.

Dataset format:
- Images: PNG files in `Data_example/images/`
- Labels: YOLO format annotations in `Data_example/labels/` (class_id, normalized_center_x, normalized_center_y, normalized_width, normalized_height)

## Models

The system supports multiple object detection models:
- RTDETR (Real-Time Detection Transformer) from Ultralytics
- YOLOv8/YOLOv11 from Ultralytics
- DINO&DDQ from MMDetection

## Contributing

Contributions to improve the codebase are welcome. Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

