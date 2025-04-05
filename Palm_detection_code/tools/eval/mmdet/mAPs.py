"""
Description:
This script evaluates object detection predictions using the COCO API.
It loads prediction results and ground truth annotations, performs evaluation
across multiple IoU thresholds, and prints a summary of the mean average precision (mAP)
values (overall and per IoU threshold) along with per-class mAP.
The evaluation is performed for both validation and test splits.
All file paths use generic placeholder paths to avoid exposing personal information.

Usage:
    python mAPs.py --methods ddq --dataset_index 0 [--debug]
"""

import os
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmengine.config import Config
import subprocess
import logging
import argparse

logger = logging.getLogger(__name__)

def get_latest_checkpoint(method: str, dataset_index: int) -> str:
    """
    Get the latest checkpoint file for the specified method and dataset index.

    Parameters:
        method (str): Method name.
        dataset_index (int): Dataset index.
    
    Returns:
        str: The checkpoint file content.
    """
    checkpoint_file = f"/path/to/your/output/{method}/dataset{dataset_index}/last_checkpoint"
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = f.read().strip()
            logger.debug(f"Read checkpoint: {checkpoint}")
            return checkpoint
    except Exception as e:
        logger.error(f"Failed to read checkpoint file: {checkpoint_file}, error: {e}")
        raise

def evaluate_maps(pred_file: str, ann_file: str, method: str, debug: bool = False) -> np.ndarray:
    """
    Evaluate prediction results using the COCO API.

    Parameters:
        pred_file (str): Path to the prediction JSON file.
        ann_file (str): Path to the ground truth annotation JSON file.
        method (str): The method identifier.
        debug (bool): If True, only evaluates the first 50 images for debugging.
    
    Returns:
        np.ndarray: Array of evaluation statistics.
    """
    logger.info(f"Evaluating {method.upper()} model:")
    
    # Load COCO format ground truth and predictions
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(pred_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Set IoU thresholds from 0.5 to 0.95 with a step of 0.05
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
    
    # Get category names and count instances per category
    categories = {cat['id']: cat['name'] for cat in coco_gt.dataset['categories']}
    class_instances = {cat_id: 0 for cat_id in categories.keys()}
    for ann in coco_gt.dataset['annotations']:
        class_instances[ann['category_id']] += 1
    
    if debug:
        logger.info("Debug mode: Evaluating only the first 50 images")
        image_ids = sorted(list(coco_gt.getImgIds()))[:50]
        coco_eval.params.imgIds = image_ids

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    precisions = coco_eval.eval['precision']
    
    # 构造表头
    header = "Class           Instances  " + " ".join([f"mAP{iou:<7}" for iou in range(50, 100, 5)]) + "mAP"
    logger.info("\n" + header)
    logger.info("-" * (len(header) + 5))
    
    total_instances = sum(class_instances.values())
    overall_line = f"{'all':<15}{total_instances:<10}"
    all_aps = []
    for iou_idx in range(10):
        mean_ap = np.mean(precisions[iou_idx, :, :, 0, -1])
        all_aps.append(mean_ap)
        overall_line += f"{mean_ap:>7.4f} "
    mean_map = np.mean(all_aps)
    overall_line += f"{mean_map:>7.4f}"
    logger.info(overall_line)
    
    # Evaluate each category
    for idx, (cat_id, cat_name) in enumerate(sorted(categories.items())):
        row = f"{cat_name:<15}{class_instances[cat_id]:<10}"
        class_aps = []
        for iou_idx in range(10):
            ap = np.mean(precisions[iou_idx, :, idx, 0, -1])
            class_aps.append(ap)
            row += f"{ap:>7.4f} "
        class_map = np.mean(class_aps)
        row += f"{class_map:>7.4f}"
        logger.info(row)
    
    # 输出总体统计信息
    logger.info("\nOverall evaluation statistics:")
    logger.info(f"Average mAP (all IoU thresholds): {mean_map:.4f}")
    logger.info(f"mAP@0.5: {all_aps[0]:.4f}")
    logger.info(f"mAP@0.75: {all_aps[5]:.4f}")
    
    return coco_eval.stats

def run_command(cmd: str, description: str) -> bool:
    """
    Run a shell command and log its description.

    Parameters:
        cmd (str): The command to run.
        description (str): Description of the command.
    
    Returns:
        bool: True if the command succeeded, False otherwise.
    """
    logger.info(f"Running {description}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        logger.error(f"Error: {description} failed with return code {result.returncode}")
        return False
    logger.debug(f"{description} completed successfully.")
    return True

def main(args: argparse.Namespace) -> None:
    """
    Main function to evaluate predictions on the validation and test sets.

    Parameters:
        args (argparse.Namespace): Command-line arguments.
        
    For each method, this function:
      - Creates a temporary directory.
      - Prepares a configuration file for the validation set and runs inference.
      - Prepares a configuration file for the test set and runs inference.
      - Evaluates the predictions using the COCO API.
      - Outputs evaluation results and file paths.
    """
    methods = args.methods
    dataset_index = args.dataset_index
    debug_mode = args.debug
    
    for method in methods:
        logger.info("\n" + "=" * 20 + f" Evaluating {method.upper()} " + "=" * 20)
        logger.info(f"Debug mode: {debug_mode}")
        
        # 创建临时输出目录
        temp_dir = f"/path/to/your/output/work_dirs/temp_eval_{method}"
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Temporary directory created at: {temp_dir}")
        
        # 准备验证集配置文件
        config_file = f"/path/to/your/configs/{method}/{method}_{dataset_index}.py"
        cfg = Config.fromfile(config_file)
        
        cfg.test_dataloader.dataset.ann_file = f"/path/to/your/dataset/COCO_splits/dataset_{dataset_index}/annotations/val.json"
        cfg.test_dataloader.dataset.data_root = f"/path/to/your/dataset/COCO_splits/dataset_{dataset_index}"
        cfg.test_dataloader.dataset.data_prefix = dict(img='val/')
        
        if debug_mode:
            cfg.test_dataloader.dataset.indices = list(range(50))
            cfg.test_dataloader.batch_size = 1
            cfg.test_dataloader.num_workers = 1
        else:
            cfg.test_dataloader.dataset.indices = None
        
        cfg.test_evaluator.format_only = True
        cfg.test_evaluator.outfile_prefix = os.path.join(temp_dir, 'val_predictions')
        
        # 保存临时验证集配置文件
        temp_config_val = os.path.join(temp_dir, 'temp_config_val.py')
        cfg.dump(temp_config_val)
        
        # 运行验证集推理命令
        val_checkpoint = get_latest_checkpoint(method, dataset_index)
        cmd_val = f'python /path/to/your/mmdetection/tools/test.py {temp_config_val} {val_checkpoint}'
        if not run_command(cmd_val, f"prediction on validation set for {method}"):
            continue
        
        # 准备测试集配置文件
        cfg.test_dataloader.dataset.ann_file = f"/path/to/your/dataset/COCO_splits/dataset_{dataset_index}/annotations/test.json"
        cfg.test_dataloader.dataset.data_prefix = dict(img='test/')
        cfg.test_evaluator.outfile_prefix = os.path.join(temp_dir, 'test_predictions')
        
        temp_config_test = os.path.join(temp_dir, 'temp_config_test.py')
        cfg.dump(temp_config_test)
        
        # 运行测试集推理命令
        test_checkpoint = get_latest_checkpoint(method, dataset_index)
        cmd_test = f'python /path/to/your/mmdetection/tools/test.py {temp_config_test} {test_checkpoint}'
        if not run_command(cmd_test, f"prediction on test set for {method}"):
            continue
        
        # 评估验证集结果
        val_pred_file = os.path.join(temp_dir, 'val_predictions.bbox.json')
        logger.info(f"\n{method.upper()} Validation Results:")
        evaluate_maps(val_pred_file,
                      f"/path/to/your/dataset/COCO_splits/dataset_{dataset_index}/annotations/val.json",
                      method, debug=debug_mode)
        
        # 评估测试集结果
        test_pred_file = os.path.join(temp_dir, 'test_predictions.bbox.json')
        logger.info(f"\n{method.upper()} Test Results:")
        evaluate_maps(test_pred_file,
                      f"/path/to/your/dataset/COCO_splits/dataset_{dataset_index}/annotations/test.json",
                      method, debug=debug_mode)
        
        logger.info(f"\nPrediction files for {method} are saved in:")
        logger.info(f"Validation predictions: {val_pred_file}")
        logger.info(f"Test predictions: {test_pred_file}")
        logger.info(f"Temporary directory for {method} is kept at: {temp_dir}")

if __name__ == '__main__':
    # 配置 logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Evaluate object detection predictions using COCO API for validation and test sets."
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=["ddq"],
        help="List of method identifiers (default: ['ddq'])"
    )
    parser.add_argument(
        "--dataset_index", type=int, default=0,
        help="Dataset index to use (default: 0)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode to evaluate only the first 50 images (default: False)"
    )
    
    args = parser.parse_args()
    main(args)

# PYTHONPATH="$PYTHONPATH:/data/general/development/kangning/DETECTION/mmdetection" CUDA_VISIBLE_DEVICES=1 python work_dirs/eval_maps.py > outputs/dino_eval.log 2>&1