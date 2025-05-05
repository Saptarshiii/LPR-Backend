import cv2
import time
from services.yolo_model import model

def generate_live_video():
    cap = cv2.VideoCapture(1)  # Default webcam

    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO and draw results
            results = model(frame)
            result_frame = results[0].plot()

            # Encode frame to JPEG
            ret, jpeg = cv2.imencode('.jpg', result_frame)
            if not ret:
                continue

            frame_bytes = jpeg.tobytes()

            # Yield MJPEG frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

            time.sleep(0.03)  # Optional: adjust frame rate
    finally:
        cap.release()
