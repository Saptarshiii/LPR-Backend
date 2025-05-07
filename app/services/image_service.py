from services.yolo_model import model
from utils.ocr_utils import extract_text_from_plate
import cv2
import numpy as np

def process_image_with_ocr(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(img)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = img[y1:y2, x1:x2]
            text = extract_text_from_plate(plate_img)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, text, (x1, max(y1 - 10, 0)),  # prevent going out of bounds
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()
