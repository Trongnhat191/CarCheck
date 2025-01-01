import argparse
from inference.models.utils import get_roboflow_model
import supervision as sv
import cv2
import numpy as np
from ultralytics import YOLO
import json
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='vi')

model_path = 'models/yolov8/detect/train/weights/best.pt'
plate_model = YOLO(model_path)

SOURCE_video2 = np.array([[123, 256],[321, 713],[917, 359],[501, 120]])
# SOURCE_video = np.array([[0, 0],[0, 1080],[1920, 1080],[1920, 0]])
def visual_bbox(
    img,  # Thay đổi từ img_path thành img
    predictions,
    conf_thres=0.5,
    font=cv2.FONT_HERSHEY_SIMPLEX
):
    # Bỏ dòng đọc ảnh vì đã có img
    # img = cv2.imread(img_path)
    
    h, w = img.shape[:2]
    for prediction in predictions[1:]:
        conf_score = prediction['confidence']
        if conf_score < conf_thres:
            continue
            
        bbox = prediction['box']
        xmin = int(bbox['x1'])
        ymin = int(bbox['y1'])
        xmax = int(bbox['x2'])
        ymax = int(bbox['y2'])
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        text = f"{conf_score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, font, 1, 2)
        cv2.rectangle(img, (xmin, ymin - text_height - 5), (xmin + text_width, ymin), (0, 255, 0), -1)
        cv2.putText(img, text, (xmin, ymin - 5), font, 1, (0, 0, 0), 2)
    return img

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a model using Roboflow's API"
    )
    parser.add_argument(
        "--source_video_path",
        required= True,
        help="Path to the video file to run inference on",
        type = str,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    # print(video_info.resolution_wh)
    # model = get_roboflow_model("yolov8n-640")

    byte_track = sv.ByteTrack(frame_rate = video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh = video_info.resolution_wh
    )
    # print(f"thickness: {thickness}")
    text_scale = sv.calculate_optimal_text_scale(resolution_wh = video_info.resolution_wh)
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness = thickness)
    label_annotator = sv.LabelAnnotator(text_scale = text_scale, text_thickness = thickness)
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(SOURCE_video2)
    for frame in frame_generator:
        result = plate_model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        detections = detections[polygon_zone.trigger(detections)]
        detections = byte_track.update_with_detections(
            detections = detections
        )

        #car bounding box
        if (len(detections.xyxy) != 0):
            car_x_min = int(detections.xyxy[0][0])
            car_y_min = int(detections.xyxy[0][1])
            car_x_max = int(detections.xyxy[0][2])
            car_y_max = int(detections.xyxy[0][3])
            # print(f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}")
            # print('---')
            #crop image
            car_crop_img = frame[car_y_min:car_y_max, car_x_min:car_x_max]
            # cv2.imshow("crop_img", crop_img)
            plate_results = plate_model(car_crop_img, verbose=False)
            plate_predictions = json.loads(plate_results[0].to_json())
            
            # plate_x_min = int(plate_predictions[1]['box']['x1'])
            # plate_y_min = int(plate_predictions[1]['box']['y1'])
            # plate_x_max = int(plate_predictions[1]['box']['x2'])
            # plate_y_max = int(plate_predictions[1]['box']['y2'])
            # plate_crop_img = car_crop_img[plate_y_min:plate_y_max, plate_x_min:plate_x_max]
            # plate_number = reader.readtext(plate_crop_img)
            # print(plate_number)
            # cv2.imshow("plate_crop_img", plate_crop_img)
            img = visual_bbox(car_crop_img, plate_predictions)
            # cv2.imshow('img', img)
            # key = cv2.waitKey(0)
        labels = [
            f"#{tracker_id}"
            for tracker_id
            in detections.tracker_id
        ]
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon = SOURCE_video2, color = sv.Color(0, 255, 0), thickness = thickness)
        annotated_frame = bounding_box_annotator.annotate(
            scene = annotated_frame,
            detections = detections
        )

        annotated_frame = label_annotator.annotate(
            scene = annotated_frame,
            detections = detections,
            labels = labels
        )
        
        cv2.imshow("annotated_frame", annotated_frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()