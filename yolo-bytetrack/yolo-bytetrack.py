from ultralytics import YOLO
import math
import time

SHOW_VIDEO = True
VIDEO_NAME = "testwalk"
YOLO_MODEL = "yolov8n.pt"

model = YOLO(f'../yolo-weights/{YOLO_MODEL}')
# TODO: Dabūt, lai darbina GPU
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

for result in results:
    boxes = result.boxes

    new_frame_time = time.time()
    frame_count += 1

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
        boxId = int(box.id)


    fps = 1 / (new_frame_time - prev_frame_time)
    fps_sum += fps
    prev_frame_time = new_frame_time

print(f'AVERAGE FPS: {fps_sum / frame_count}')
