from roboflow import Roboflow
import torchvision
import os
from transformers import AutoFeatureExtractor

import torchvision
from torchvision.transforms import ToTensor
import subprocess

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("asusevski").project("dota-llm")
dataset = project.version(2).download("yolov8")

subprocess.run(["yolo", "task=detect", "mode=train", "model=yolov8s.pt", f"data={dataset.location}", "epochs=50", "imgsz=640"])