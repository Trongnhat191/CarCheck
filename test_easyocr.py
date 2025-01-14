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
model = YOLO('yolov8n.pt')#for car
model1 = YOLO(model_path)#for plate

img_path = 'cars/car132.png'
img = cv2.imread(img_path)
cv2.imshow('img', img)

results = model(img_path, verbose=False)
# print(results)
# predictions = json.loads(results[0].to_json())
# # print(predictions)
# car_x_min = int(predictions[0]['box']['x1'])
# car_y_min = int(predictions[0]['box']['y1'])
# car_x_max = int(predictions[0]['box']['x2'])
# car_y_max = int(predictions[0]['box']['y2'])
# cv2.imshow('car', img[car_y_min:car_y_max, car_x_min:car_x_max])
# print(plate_x_min, plate_y_min, plate_x_max, plate_y_max)
# car_img = img[car_y_min:car_y_max, car_x_min:car_x_max]

res_plate = model1(img)
predictions_plate = json.loads(res_plate[0].to_json())
print(predictions_plate)
plate_x_min = int(predictions_plate[1]['box']['x1'])-1
plate_y_min = int(predictions_plate[1]['box']['y1'])-1
plate_x_max = int(predictions_plate[1]['box']['x2'])+1
plate_y_max = int(predictions_plate[1]['box']['y2'])+1

plate_img = img[plate_y_min:plate_y_max, plate_x_min:plate_x_max]
cv2.imshow('plate', plate_img)



import cv2
import numpy as np

def increase_resolution(plate_img, scale_factor=2):
    # Tăng kích thước ảnh bằng super resolution
    # Sử dụng INTER_CUBIC cho kết quả tốt với text
    high_res = cv2.resize(plate_img, None, 
                         fx=scale_factor, 
                         fy=scale_factor, 
                         interpolation=cv2.INTER_CUBIC)
    
    return high_res

# Sử dụng hàm với scale_factor=2 để tăng gấp đôi độ phân giải
high_res_plate = increase_resolution(plate_img, scale_factor=2)

# Hiển thị kết quả
cv2.imshow('Original Plate', plate_img)
cv2.imshow('High Resolution Plate', high_res_plate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cập nhật biến plate_img với ảnh độ phân giải cao
plate_img = high_res_plate
res = ocr.ocr(plate_img)
print(res)
# plate_num =  res[0][0][1][0]
# print(plate_num)
# cv2.imshow('plate_img', plate_img)
# img = visual_bbox(img_path, predictions, plate_num)
# cv2.imshow('img', img)
key = cv2.waitKey(0)
cv2.destroyAllWindows() 