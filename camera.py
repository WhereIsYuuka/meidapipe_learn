import cv2
import time
from pathlib import Path
# from ultralytics import YOLO

# 加载模型
# model = YOLO("yolov8n-pose.pt")

camera = cv2.VideoCapture(0)
last_saved_time = 0

while camera.isOpened():
    success, image = camera.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    # 检测
    current_time = time.time()
    if current_time - last_saved_time >= 0.5:  # 每半秒记录一次
        last_saved_time = current_time
        # results = model(source=image, show=True, conf=0.3, save=True)
        # cv2.imwrite(f"../output/{current_time}.jpg", image)
        # print(f"Save image to ../output/{current_time}.jpg")
        timestamp = int(round(current_time * 1000))
        path = Path("D:\Code\Python\jpg_path")
        path.mkdir(parents=True, exist_ok=True)
        path_image = path / f"{timestamp}.jpg"
        cv2.imwrite(str(path_image), image)

        print(f"Save image to {path_image}")

    # 显示图像
    cv2.imshow('YOLO Detection with Keypoints', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()