import subprocess

command = ["python",
"/deac/csc/paucaGrp/zhur/mmdetection/mmdetection/tools/train.py",
"/deac/csc/paucaGrp/zhur/mmdetection/mmdetection/configs/dino/dino_1.py",
"--work-dir", "/deac/csc/paucaGrp/zhur/mmdetection/mmdetection/_Palm_output/dino/dataset1"
]

subprocess.run(command)