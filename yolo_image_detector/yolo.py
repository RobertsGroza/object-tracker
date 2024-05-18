from ultralytics import YOLO
import cv2

model = YOLO('../yolo_weights/yolov8l.pt')
results = model("kakis.jpg", show=True)
cv2.waitKey(0)
