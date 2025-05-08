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

            # Draw green bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate placeholder size for text
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text_width, text_height = text_size
            padding = 10
            rect_x1 = x1
            rect_y1 = max(y1 - text_height - 2 * padding, 0)
            rect_x2 = x1 + text_width + 2 * padding
            rect_y2 = rect_y1 + text_height + 2 * padding

            # Draw blue rectangle placeholder
            cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), thickness=cv2.FILLED)

            # Draw white text over blue placeholder
            text_x = rect_x1 + padding
            text_y = rect_y1 + padding + text_height
            cv2.putText(img, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    _, buffer = cv2.imencode('.png', img)
    return buffer.tobytes()
