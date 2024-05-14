import os
import cv2

video_path = os.path.join('.', 'videos', 'riteni.mp4')
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
frame_count = 0

while success:
    frame_count += 1
    # cv2.imshow("Image", frame)
    # cv2.waitKey(1)
    success, frame = cap.read()

print("Frame count: " + str(frame_count))
cap.release()
cv2.destroyAllWindows()
