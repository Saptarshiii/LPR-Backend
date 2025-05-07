import cv2
import numpy as np
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    return edged

def extract_text_from_plate(plate_img):
    processed = preprocess_plate(plate_img)
    text = pytesseract.image_to_string(processed, config='--psm 8')
    return text.strip()
