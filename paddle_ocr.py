import cv2
from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(lang = 'vi')
image = cv2.imread('car1.jpeg')
res = ocr.ocr(image)

#show res
for line in res:
    print(line)
    