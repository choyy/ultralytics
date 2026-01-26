import os
from ultralytics import YOLO
from pathlib import Path

model_name = "runs/obb/train9/weights/best.pt"
file_name = "ultralytics/cfg/datasets/ellipse-line-obb.yaml"

model = YOLO(model_name)
model.eval()

onnx_path = model.export(format='onnx', simplify=True, device="0")

save_name = f"{Path(model_name).stem}"
os.rename(onnx_path, os.path.join(f'{save_name}.onnx'))

