from ultralytics import YOLO
import math
import time
import cv2
import json
from classnames import class_names

SHOW_VIDEO = True
VIDEO_NAME = "testwalk"
YOLO_MODEL = "yolov8n.pt"

model = YOLO(f'../yolo-weights/{YOLO_MODEL}')
results = model.track(
    f'../../object-tracker-shared/videos/{VIDEO_NAME}.mp4',
    show=SHOW_VIDEO,
    stream=True,
    tracker="bytetrack.yaml",
)

prev_frame_time = 0
new_frame_time = 0
frame_count = 0
fps_sum = 0

# Use videoCapture purely to get FrameRate
cap = cv2.VideoCapture(f'../../object-tracker-shared/videos/{VIDEO_NAME}.mp4')
video_frame_rate = cap.get(cv2.CAP_PROP_FPS)
out_file_path = f'../../object-tracker-shared/bytetrack-outputs/{VIDEO_NAME}.txt'
out_file = open(out_file_path, "w")
distinct_ids = []

for result in results:
    boxes = result.boxes

    new_frame_time = time.time()
    frame_count += 1

    detections = []

    for box in boxes:
        # Bounding Box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        # Class Name
        cls = int(box.cls[0])
        # Confidence score
        conf = math.ceil((box.conf[0] * 100)) / 100
        # ID
        object_id = int(box.id)
        object_class = class_names[cls]

        if not any(el["id"] == object_id for el in distinct_ids):
            distinct_ids.append({"id": object_id, "class": object_class})

        detections.append(json.dumps({
            "id": object_id,
            "class": object_class,
            "x": x1,
            "y": y1,
            "width": w,
            "height": h,
        }))

    fps = 1 / (new_frame_time - prev_frame_time)
    fps_sum += fps
    prev_frame_time = new_frame_time
    out_file.write(f'[{", ".join(detections)}]\n')

print(f'AVERAGE FPS: {fps_sum / frame_count}')

# Prepend summary to output file
out_file.close()
out_file = open(out_file_path, "r")
content = out_file.read()
out_file_w_summary = open(out_file_path, "w")
out_file_w_summary.write(f'{json.dumps({"fps": video_frame_rate, "ids": distinct_ids})}\n')
out_file_w_summary.write(content)

cap.release()
out_file.close()
out_file_w_summary.close()
