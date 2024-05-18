import cv2
import cvzone
import math
import json
import sys
from ultralytics import YOLO
from deep_sort.tracker import Tracker
from classnames import class_names
from fps_counter import FPSCounter
from video_summary import VideoSummary

VIDEO_NAME = sys.argv[1]
SHOW_VIDEO = sys.argv[2] == "true" if len(sys.argv) > 2 else False

model = YOLO("yolo_weights/yolov8l.pt")
cap = cv2.VideoCapture(f'../object-tracker-shared/videos/{VIDEO_NAME}.mp4')
out_file_path = f'../object-tracker-shared/deepsort-outputs/{VIDEO_NAME}.txt'
out_file = open(out_file_path, "w")

video_summary = VideoSummary(cap.get(cv2.CAP_PROP_FPS))
frame_counter = FPSCounter()
tracker = Tracker()
success, frame = cap.read()

while success:
    results = model(frame)
    detections = []
    detections_for_tracking = []
    frame_counter.process_new_frame()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            object_class = int(box.cls[0])
            confidence = math.ceil((box.conf[0] * 100)) / 100
            detections_for_tracking.append([x1, y1, x2, y2, confidence, object_class])

        tracker.update(frame, detections_for_tracking)
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width, height = x2 - x1, y2 - y1
            track_id = track.track_id
            track_class = track.class_id
            video_summary.add_object(track_id, track_class)
            detections.append(json.dumps({
                "id": track_id,
                "class": class_names[track_class],
                "x": x1,
                "y": y1,
                "width": width,
                "height": height,
            }))

            # Draw object bounding boxes
            if SHOW_VIDEO:
                cvzone.cornerRect(frame, (x1, y1, width, height), l=10, t=3)
                cvzone.putTextRect(
                    frame,
                    f'{class_names[track_class]}, {track_id}', (max(0, x1), max(35, y1)),
                    scale=1, offset=2, thickness=1
                )

    cv2.imshow("Image", frame)
    cv2.waitKey(1)
    out_file.write(f'[{", ".join(detections)}]\n')
    success, frame = cap.read()

out_file.close()
video_summary.prepend_summary_to_file(out_file_path)
frame_counter.print_average_fps()

cap.release()
cv2.destroyAllWindows()
