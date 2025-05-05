from fastapi import UploadFile
from PIL import Image
import io
import numpy as np
import cv2
from services.yolo_model import model

async def process_image(file: UploadFile) -> bytes:
    # Load image
    image = Image.open(file.file).convert("RGB")
    np_img = np.array(image)

    # Run YOLO inference
    results = model(np_img)

    # Draw bounding boxes on image
    result_img = results[0].plot()  # draws boxes on image

    # Convert and return as bytes
    _, buffer = cv2.imencode(".png", result_img)
    return buffer.tobytes()
