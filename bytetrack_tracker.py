from ultralytics import YOLO
import cv2
import json
import sys
from classnames import class_names
from fps_counter import FPSCounter
from video_summary import VideoSummary

VIDEO_NAME = sys.argv[1]
SHOW_VIDEO = sys.argv[2] == "true" if len(sys.argv) > 2 else False

video_path = f'../object-tracker-shared/videos/{VIDEO_NAME}.mp4'
model = YOLO("yolo_weights/yolov8l.pt")
results = model.track(video_path, show=SHOW_VIDEO, stream=True, tracker="bytetrack.yaml")

# Use VideoCapture to get original video frame rate
cap = cv2.VideoCapture(video_path)
video_summary = VideoSummary(cap.get(cv2.CAP_PROP_FPS))
cap.release()

out_file_path = f'../object-tracker-shared/bytetrack-outputs/{VIDEO_NAME}.txt'
out_file = open(out_file_path, "w")
frame_counter = FPSCounter()

for result in results:
    boxes = result.boxes
    detections = []
    frame_counter.process_new_frame()

    for box in boxes:
        if box.id is not None:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            object_id = int(box.id)
            object_class = class_names[int(box.cls[0])]
            video_summary.add_object(object_id, object_class)
            detections.append(json.dumps({
                "id": object_id,
                "class": object_class,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            }))

    out_file.write(f'[{", ".join(detections)}]\n')

out_file.close()
video_summary.prepend_summary_to_file(out_file_path)
frame_counter.print_average_fps()
