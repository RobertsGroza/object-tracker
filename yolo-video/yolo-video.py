import numpy as np
from ultralytics import YOLO
from sort import *
import cv2
import cvzone
import math
import time
import json

class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

VIDEO_NAME = "testwalk"
SHOW_VIDEO = False
SHOW_YOLO_BOUNDING_BOXES = False
SHOW_TRACKER_BOUNDING_BOXES = False
CONFIDENCE_THRESHOLD = 0.5
YOLO_MODEL = "yolov8n"

model = YOLO(f'../yolo-weights/{YOLO_MODEL}.pt')
tracker = Sort(max_age=15, min_hits=2, iou_threshold=0.3)
cap = cv2.VideoCapture(f'../../object-tracker-shared/videos/{VIDEO_NAME}.mp4')
out_file_path = f'../../object-tracker-shared/outputs/{VIDEO_NAME}.txt'
out_file = open(out_file_path, "w")

prev_frame_time = 0
new_frame_time = 0
success, img = cap.read()
frame_count = 0
fps_sum = 0
distinct_classes = []
distinct_ids = []
video_frame_rate = cap.get(cv2.CAP_PROP_FPS)

while success:
    results = model(img, stream=True)
    detections = []
    new_frame_time = time.time()
    frame_count += 1

    # TODO: Fix doubling detections
    detections_for_tracking = np.empty((0, 6))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Class Name
            cls = int(box.cls[0])
            # Confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100

            if conf < CONFIDENCE_THRESHOLD:
                continue

            current_array = np.array([int(x1), y1, x2, y2, conf, cls])
            detections_for_tracking = np.vstack((detections_for_tracking, current_array))

            if SHOW_VIDEO and SHOW_YOLO_BOUNDING_BOXES:
                cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=3)
                cvzone.putTextRect(
                    img, f'{class_names[cls]} {conf}', (max(0, x1), max(35, y1)),
                    scale=1, offset=2, thickness=1
                )

    # Tracker results
    tracker_results = tracker.update(detections_for_tracking)
    for result in tracker_results:
        x1, y1, x2, y2, object_id, class_id = result
        x1, y1, x2, y2, object_id, class_id = int(x1), int(y1), int(x2), int(y2), int(object_id), int(class_id)
        width, height = x2 - x1, y2 - y1
        object_class = class_names[int(class_id)]

        if not any(el["id"] == object_id for el in distinct_ids):
            distinct_ids.append({"id": object_id, "class": object_class})

        detections.append(json.dumps({
            "id": object_id,
            "class": object_class,
            "x": x1,
            "y": y1,
            "width": width,
            "height": height,
        }))

        if SHOW_VIDEO and SHOW_TRACKER_BOUNDING_BOXES:
            cvzone.cornerRect(img, (x1, y1, width, height), l=10, t=3)
            cvzone.putTextRect(
                img, f'{object_class}, ID: {int(object_id)}', (max(0, x1), max(35, y1)),
                scale=1, offset=2, thickness=1
            )

    fps = 1 / (new_frame_time - prev_frame_time)
    fps_sum += fps
    prev_frame_time = new_frame_time
    out_file.write(f'[{", ".join(detections)}]\n')

    if SHOW_VIDEO:
        cvzone.putTextRect(img, f'{math.floor(fps)} FPS', (10, 15), scale=1, thickness=1, colorR=0, colorB=0)
        cv2.imshow("Image", img)

    cv2.waitKey(1)
    success, img = cap.read()

print(f'Video FPS: {video_frame_rate}')
print(f'AVERAGE FPS: {fps_sum / frame_count}')

# Prepend summary to output file
out_file.close()
out_file = open(out_file_path, "r")
content = out_file.read()
out_file_w_summary = open(out_file_path, "w")
out_file_w_summary.write(f'{json.dumps({"fps": video_frame_rate, "ids": distinct_ids})}\n')
out_file_w_summary.write(content)

# Release memory
out_file.close()
out_file_w_summary.close()
cap.release()
cv2.destroyAllWindows()
