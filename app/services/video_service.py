from fastapi import UploadFile
import cv2
import tempfile
import os
from services.yolo_model import model

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

        # Run YOLO detection and draw boxes
        results = model(frame)
        result_frame = results[0].plot()
        out.write(result_frame)

    cap.release()
    out.release()

    # Read output video bytes
    with open(output_path, "rb") as f:
        output_bytes = f.read()

    os.remove(input_path)
    os.remove(output_path)

    return output_bytes
