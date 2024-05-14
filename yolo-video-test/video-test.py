from ultralytics import YOLO
import cv2
import cvzone
import math
import time

model = YOLO(f'../yolo-weights/yolov8x.pt')
cap = cv2.VideoCapture(f'../../object-tracker-shared/videos/testwalk.mp4')

prev_frame_time = 0
new_frame_time = 0
success, img = cap.read()
frame_count = 0
fps_sum = 0

while success:
    results = model(img, stream=True)
    detections = []
    new_frame_time = time.time()
    frame_count += 1

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
            cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=3)
            cvzone.putTextRect(
                img, f'class: {cls} conf: {conf}', (max(0, x1), max(35, y1)),
                scale=1, offset=2, thickness=1
            )

    fps = 1 / (new_frame_time - prev_frame_time)
    fps_sum += fps
    prev_frame_time = new_frame_time
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    success, img = cap.read()

print(f'AVERAGE FPS: {fps_sum / frame_count}')

cap.release()
cv2.destroyAllWindows()
