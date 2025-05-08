import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    ret, binary_image = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)

    # Reduce 10 pixels from all sides
    h, w = binary_image.shape
    cropped = binary_image[10:h-10, 10:w-10]

    # Display the cropped and preprocessed image
    # cv2.imshow("Preprocessed Plate", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return cropped


def extract_text_from_plate(plate_img):
    processed = preprocess_plate(plate_img)
    text = pytesseract.image_to_string(processed, config='--psm 8')
    return text.strip()


# from PIL import Image
# import pytesseract
# import numpy as np
# 

# def extract_text_from_plate(image):
#     if image is None:
#         return ""

#     try:
#         if isinstance(image, np.ndarray):
#             image = Image.fromarray(image)

#         if image.width == 0 or image.height == 0:
#             return ""

#         return pytesseract.image_to_string(image, config='--psm 8')
#     except Exception as e:
#         print(f"OCR error: {e}")
#         return ""
