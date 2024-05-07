from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import json

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

videoName = "riteni"
SHOW_VIDEO = False
model = YOLO("../yolo-weights/yolov8n.pt")

out_file = open(f'./outputs/{videoName}.txt', "w")
cap = cv2.VideoCapture(f'./videos/{videoName}.mp4')  # For Video

prev_frame_time = 0
new_frame_time = 0
success, img = cap.read()


while success:
    new_frame_time = time.time()
    results = model(img, stream=True)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            if SHOW_VIDEO:
                cvzone.cornerRect(img, (x1, y1, w, h), l=10, t=3)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if SHOW_VIDEO:
                cvzone.putTextRect(
                    img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),
                    scale=1, offset=2, thickness=1
                )

            # TODO: Izfiltrēt rezultātus ar zemu confidence
            detectedObject = {
                "id": 1,
                "class": classNames[cls],
                "x": x1,
                "y": y1,
                "width": w,
                "height": h,
            }
            detections.append(json.dumps(detectedObject))


    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    if SHOW_VIDEO:
        cvzone.putTextRect(img, f'{math.floor(fps)} FPS', (10, 15), scale=1, thickness=1, colorR=0, colorB=0)
    print(f'{math.floor(fps)} FPS')

    # TODO: Varbūt izdrukāt vidējo FPS, priekš salīdzināšanas darbā
    out_file.write(f'[{", ".join(detections)}]\n')

    if SHOW_VIDEO:
        cv2.imshow("Image", img)
    cv2.waitKey(1)
    success, img = cap.read()

out_file.close()
cap.release()
cv2.destroyAllWindows()
