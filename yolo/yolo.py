from ultralytics import YOLO
import cv2

model = YOLO('../yolo-weights/yolov8l.pt')
results = model("kakis.jpg", show=True, save=True)
cv2.waitKey(0)
