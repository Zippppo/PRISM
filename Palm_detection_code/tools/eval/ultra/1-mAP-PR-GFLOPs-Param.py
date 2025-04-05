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


ultralytics.checks()

def eval(model_name=None, epochs=None, dataset_idx=None):

    pt_path = f"/deac/csc/alqahtaniGrp/cuij/yolo/cali_result/{model_name}/dataset_{dataset_idx}/{epochs}epochs/weights/best.pt"
    save_path = f"/deac/csc/alqahtaniGrp/cuij/yolo/cali_result/{model_name}/evalue_test_iou_param/dataset_{dataset_idx}"

    os.makedirs(save_path, exist_ok=True)
    DEFAULT_CFG.save_dir = save_path

    # try:
    model = RTDETR(pt_path)
    metrics = model.val(iou=0.5)

    print("map50-95:", metrics.box.map)
    print("map50:", metrics.box.map50)
    print("map75:", metrics.box.map75)
    print("map50-95 of each category:", metrics.box.maps)
    print("Recall", metrics.box.r)
    print("Precision", metrics.box.p)


    # Save performance metrics to a CSV file
    csv_path = os.path.join(save_path, f"performance-{epochs}epochs.csv")

    with open(csv_path, 'a') as f:
        # f.write(f"Model: {model}\n")
        f.write(f"map50-95: {metrics.box.map}\n")
        f.write(f"map50: {metrics.box.map50}\n")
        f.write(f"map75: {metrics.box.map75}\n")
        f.write(f"map50-95 of each category: {metrics.box.maps}\n")
        f.write(f"Recall: {metrics.box.r}\n")
        f.write(f"Precision: {metrics.box.p}\n")
        f.write("\n")
    print("Performance metrics saved to:", csv_path)
    return True
    
    # except Exception as e:
    #     print("Failed{dataset_{dataset_idx}, {epochs}epochs}:{str(e)")
    #     return False

model_name = "RTDETR"
epochs_list = [100]
dataset_indices = [0,1,2,3,4]

summary_path = f"/deac/csc/alqahtaniGrp/cuij/yolo/cali_result/{model_name}/evl_sum.txt"

with open(summary_path, 'w') as f:
    f.write("评估时间：{time.strftime('%y-%m-%d %H:%M:%S)}\n")
    f.write("="*50 + "\n\n")

for dataset_idx in dataset_indices:
    for epochs in epochs_list:
        print(f"\n开始评估 dataset_{dataset_idx}, {epochs}epochs")
        success = eval(model_name, epochs, dataset_idx)

        with open(summary_path, 'a') as f:
            status = "success" if success else "failed"
            f.write(f"dataset_{dataset_idx}, {epochs}epochs: {status}\n")
print(f"Done! saved to {summary_path}")