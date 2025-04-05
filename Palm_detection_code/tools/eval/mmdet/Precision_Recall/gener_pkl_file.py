import subprocess

command = ["python",
"//deac/csc/paucaGrp/zhur/mmdetection/mmdetection/tools/test.py",
"/deac/csc/paucaGrp/zhur/mmdetection/mmdetection/configs/dino/dino_0.py",
"/deac/csc/paucaGrp/zhur/mmdetection/mmdetection/_Palm_output/dino/dataset0/best_coco_bbox_mAP_epoch_30.pth",
"--out",
"/deac/csc/paucaGrp/zhur/mmdetection/pkl_files/dino-0.pkl"
]

subprocess.run(command)