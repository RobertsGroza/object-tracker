import cv2
import cvzone
import math
import time
import json
from ultralytics import YOLO
from tracker import Tracker
from classnames import class_names

VIDEO_NAME = "testwalk"
YOLO_MODEL = "yolov8n"
SHOW_VIDEO = True

cap = cv2.VideoCapture(f'../../object-tracker-shared/videos/{VIDEO_NAME}.mp4')
out_file_path = f'../../object-tracker-shared/deepsort-outputs/{VIDEO_NAME}.txt'
model = YOLO(f'../yolo-weights/{YOLO_MODEL}.pt')
out_file = open(out_file_path, "w")

prev_frame_time = 0
new_frame_time = 0
frame_count = 0
fps_sum = 0

distinct_classes = []
distinct_ids = []
video_frame_rate = cap.get(cv2.CAP_PROP_FPS)

success, frame = cap.read()
tracker = Tracker()

while success:
    results = model(frame, stream=True)
    new_frame_time = time.time()
    frame_count += 1

    detections = []
    detections_for_tracking = []

    for r in results:
        for box in r.boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Class Name
            cls = int(box.cls[0])
            # Confidence score
            conf = math.ceil((box.conf[0] * 100)) / 100
            detections_for_tracking.append([x1, y1, x2, y2, conf, cls])

        tracker.update(frame, detections_for_tracking)
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            track_id = track.track_id
            class_id = track.class_id

            if not any(el["id"] == track_id for el in distinct_ids):
                distinct_ids.append({"id": track_id, "class": class_names[class_id]})

            detections.append(json.dumps({
                "id": track_id,
                "class": class_names[class_id],
                "x": x1,
                "y": y1,
                "width": w,
                "height": h,
            }))

            if SHOW_VIDEO:
                cvzone.cornerRect(frame, (x1, y1, w, h), l=10, t=3)
                cvzone.putTextRect(
                    frame, f'{class_names[class_id]}, {track_id}', (max(0, x1), max(35, y1)),
                    scale=1, offset=2, thickness=1
                )

    fps = 1 / (new_frame_time - prev_frame_time)
    fps_sum += fps
    prev_frame_time = new_frame_time
    out_file.write(f'[{", ".join(detections)}]\n')

    if SHOW_VIDEO:
        cvzone.putTextRect(frame, f'{math.floor(fps)} FPS', (10, 15), scale=1, thickness=1, colorR=0, colorB=0)
        cv2.imshow("Image", frame)

    cv2.waitKey(1)
    success, frame = cap.read()

print(f'Video FPS: {video_frame_rate}')
print(f'AVERAGE FPS: {fps_sum / frame_count}')

# TODO: Can create separate function??
# Prepend summary to output file
out_file.close()
out_file = open(out_file_path, "r")
content = out_file.read()
out_file_w_summary = open(out_file_path, "w")
out_file_w_summary.write(f'{json.dumps({"fps": video_frame_rate, "ids": distinct_ids})}\n')
out_file_w_summary.write(content)

cap.release()
cv2.destroyAllWindows()
