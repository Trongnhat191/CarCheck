import argparse
import supervision as sv
from ultralytics import YOLO
import cv2 
import numpy as np
from paddleocr import PaddleOCR
import json

def increase_resolution(plate_img, scale_factor=2):
    # Tăng kích thước ảnh bằng super resolution
    # Sử dụng INTER_CUBIC cho kết quả tốt với text
    high_res = cv2.resize(plate_img, None, 
                         fx=scale_factor, 
                         fy=scale_factor, 
                         interpolation=cv2.INTER_CUBIC)
    
    return high_res

SOURCE_video = np.array([[123, 256],[321, 713],[917, 359],[501, 120]])
ocr = PaddleOCR(lang='vi', show_log = False)
plate_nums = []
def parse_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervision of a process",
    )
    parser.add_argument(
        "--source_video_path",
        required= True,
        help="Path to the source video file",
        type = str,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_argument()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    model_path = 'models/yolov8/detect/train/weights/best.pt'
    plate_model = YOLO(model_path)
    car_model = YOLO('yolov8n.pt')

    byte_track = sv.ByteTrack(frame_rate = video_info.fps)

    thickness = 2
    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh = video_info.resolution_wh)

    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness = 2)
    label_annotator = sv.LabelAnnotator(text_scale = text_scale)

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(
        SOURCE_video,
        # frame_resolution_wh=video_info.resolution_wh
    )
    for frame in frame_generator:
        result = car_model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        try:
            # print('-----')
            # print(detections.xyxy[0])
            # print(f"x_min: {int(detections.xyxy[0][0])}, y_min: {int(detections.xyxy[0][1])}, x_max: {int(detections.xyxy[0][2])}, y_max: {int(detections.xyxy[0][3])}")
            # print('-----')
            car_frame = frame[int(detections.xyxy[0][1]):int(detections.xyxy[0][3]), int(detections.xyxy[0][0]):int(detections.xyxy[0][2])]
            cv2.imshow('car', car_frame)
            #save car image
            # cv2.imwrite(f'cars/car{car_idx}.png', car_frame)
            plate_result = plate_model(car_frame)
            predictions_plate = json.loads(plate_result[0].to_json())
            plate_x_min = int(predictions_plate[1]['box']['x1'])
            plate_y_min = int(predictions_plate[1]['box']['y1'])
            plate_x_max = int(predictions_plate[1]['box']['x2'])
            plate_y_max = int(predictions_plate[1]['box']['y2'])
            plate_img = car_frame[plate_y_min:plate_y_max, plate_x_min:plate_x_max]
            #increase resolution
            high_res_plate = increase_resolution(plate_img, scale_factor=2)
            plate_img = high_res_plate

            cv2.imshow('plate', plate_img)
            # cv2.imshow('plate', car_frame[plate_y_min:plate_y_max, plate_x_min:plate_x_max])
            # print(f"x_min: {plate_x_min}, y_min: {plate_y_min}, x_max: {plate_x_max}, y_max: {plate_y_max}")
            # plate_crop_img = car_frame[plate_y_min:plate_y_max, plate_x_min:plate_x_max]
            plate_num = ocr.ocr(plate_img)
            print(plate_num)
            # if plate_num != [None]:
            plate_nums.append(plate_num[0][0][1][0])

            # print(plate_detections.data['class_name'][1])
        except:
            pass
        detections = byte_track.update_with_detections(detections=detections)

        labels = [
            f"#{tracker_id}"
            for tracker_id
            in detections.tracker_id
        ]
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon = SOURCE_video, color = sv.Color(255, 0, 0), thickness = thickness)
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels = labels
        )

        cv2.imshow("name",annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    print(len(plate_nums))
    print(plate_nums)