import os
import cv2
import json
import matplotlib.pyplot as plt
from ultralytics import YOLO
import easyocr
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='vi', show_log = False)

def visual_bbox(
        img_path, predictions,plate_num,
        conf_thres = 0.5,
        font = cv2.FONT_HERSHEY_SIMPLEX
):
    img = cv2.imread(img_path)
    h,w = img.shape[:2]

    for prediction in predictions:
        conf_score = prediction['confidence']
        if conf_score < conf_thres:
            continue

        bbox = prediction['box']
        xmin = int(bbox['x1'])
        ymin = int(bbox['y1'])
        xmax = int(bbox['x2'])
        ymax = int(bbox['y2'])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        text = f"{plate_num}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)

        cv2.rectangle(img, (xmin, ymin - text_height - 5), (xmin + text_width, ymin), (0, 255, 0), -1)
        cv2.putText(img, text, (xmin, ymin - 5), font, 1, (0, 0, 0),2)

    return img

model_path = 'models/yolov8/detect/train/weights/best.pt'
model = YOLO(model_path)

img_path = 'images/car6.png'
img = cv2.imread(img_path)
results = model(img_path, verbose=False)
# print(results)
predictions = json.loads(results[0].to_json())
# print(predictions[1])
plate_x_min = int(predictions[1]['box']['x1'])-5
plate_y_min = int(predictions[1]['box']['y1'])-5
plate_x_max = int(predictions[1]['box']['x2'])+5
plate_y_max = int(predictions[1]['box']['y2'])+5
cv2.imshow('plate', img[plate_y_min:plate_y_max, plate_x_min:plate_x_max])
# print(plate_x_min, plate_y_min, plate_x_max, plate_y_max)
plate_img = img[plate_y_min:plate_y_max, plate_x_min:plate_x_max]
res = ocr.ocr(plate_img)
plate_num =  res[0][0][1][0]
print(plate_num)
# cv2.imshow('plate_img', plate_img)
img = visual_bbox(img_path, predictions, plate_num)
cv2.imshow('img', img)
key = cv2.waitKey(0)
cv2.destroyAllWindows() 