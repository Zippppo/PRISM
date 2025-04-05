import subprocess

command = ["python",
"//deac/csc/paucaGrp/zhur/mmdetection/mmdetection/tools/analysis_tools/confusion_matrix.py",
"/deac/csc/paucaGrp/zhur/mmdetection/mmdetection/configs/dino/dino_0.py",
"/deac/csc/paucaGrp/zhur/mmdetection/eval_output/pkl_files/dino-0.pkl",
"/deac/csc/paucaGrp/zhur/mmdetection/eval_output/precision/dino-0"
]

subprocess.run(command)