from fastapi import UploadFile
import cv2
import tempfile
import os
import numpy as np
from services.yolo_model import model
from utils.ocr_utils import extract_text_from_plate

async def process_video(file: UploadFile) -> bytes:
    # Save uploaded file to temp file
    input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Prepare output path
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # OpenCV video reading and writing
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Crop plate
                plate_img = frame[y1:y2, x1:x2]

                # OCR
                text = extract_text_from_plate(plate_img)

                # Draw bounding box and OCR text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Read output video bytes
    with open(output_path, "rb") as f:
        output_bytes = f.read()

    os.remove(input_path)
    os.remove(output_path)

    return output_bytes
