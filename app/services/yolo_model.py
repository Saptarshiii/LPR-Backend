from ultralytics import YOLO
import os
model_path = "../app/ai/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = YOLO(model_path)


