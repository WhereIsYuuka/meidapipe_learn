import glob
import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import json
import time
import numpy as np
import sys
import signal

# 处理SIGTERM信号
signal.signal(signal.SIGTERM, quit)
# 处理SIGINT信号
signal.signal(signal.SIGINT, quit)

# 初始化模型
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
model_path = Path("./yolov8m.pt")
model = YOLO(model_path)  

pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6)

last_saved_time = time.time()
delete_time = time.time()


class Watcher:
    DIRECTORY_TO_WATCH = Path("./picture")

    def __init__(self):
        self.observer = Observer()
        last_saved_time = time.time()

    def run(self):
        event_handler = Handler(self)
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()
        quit()

class Handler(FileSystemEventHandler):
    def __init__(self, watcher):
        self.watcher = watcher

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        # elif event.event_type == 'created':
        elif event.src_path.endswith(".jpg"):
            # 当检测到新图片时，处理图片
            print(f"Received created event - {event.src_path}.")
            process_image(event.src_path)

def process_image(image_path):
    attempts = 0
    while not os.path.exists(image_path) and attempts < 3:
        print(f"Waiting for file: {image_path}")
        time.sleep(1)  # 等待1秒
        attempts += 1

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
    
    global last_saved_time
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    print(f"Processing image {image_path}")
    # 使用YOLOv8检测人物
    results = model(source=image, show=False, conf=0.4)  # 提高YOLOv8的检测置信度
    print("model run")

    # 只保留检测到的人的结果
    person_results = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # 类别0表示“person”
                person_results.append(box)

    landmarks_data = []  # 存储所有人的关键点数据

    for box in person_results:
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())  # 将张量转换为NumPy数组并展平

        # 提取人物区域
        person_image = image[y1:y2, x1:x2]

        # 将图像从BGR格式转换为RGB格式
        image_rgb = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)

        # 进行姿势检测
        result_pose = pose.process(image_rgb)

        # 保存每个人的关键点数据
        if result_pose.pose_landmarks:
            person_landmarks = {
                'id': len(landmarks_data),  # 使用列表的长度作为ID
                'landmarks': []
            }
            for landmark in result_pose.pose_landmarks.landmark:
                cx = int(landmark.x * (x2 - x1)) + x1
                cy = int(landmark.y * (y2 - y1)) + y1
                cz = landmark.z
                person_landmarks['landmarks'].append({
                    'x': cx,
                    'y': cy,
                    'z': cz
                })
            landmarks_data.append(person_landmarks)

            # 将关键点重新映射回原图
            for landmark in person_landmarks['landmarks']:
                cv2.circle(image, (landmark['x'], landmark['y']), 5, (0, 255, 0), -1)

            # 绘制检测到的姿势
            mp_drawing.draw_landmarks(
                image[y1:y2, x1:x2], result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    current_time = time.time()
    if current_time - last_saved_time >= 0.5:  # 每半秒记录一次
        last_saved_time = current_time
        timestamp = int(round(current_time * 1000))
        filename = f"./output/landmarks_{timestamp}.json"
        save_landmarks(landmarks_data, filename)
        print(f"Saved landmarks data to {filename}")
    delete_time_count()



def delete_time_count():
    global delete_time
    current_time = time.time()
    if current_time - delete_time >= 10:  # 每半秒记录一次
        delete_time = current_time
        # 删除所有的图片
        delete_image()

def delete_image():
    for filename in glob.glob("./picture/*.jpg"):
        os.remove(filename)
    print("All images have been deleted.")

    # 删除所有生成的json文件
    for filename in glob.glob("./output/landmarks_*.json"):
        os.remove(filename)
    print("All generated JSON files have been deleted.")

def save_landmarks(landmarks, filename):
    with open(filename, 'w') as f:
        json.dump(landmarks, f)

def quit():
    cv2.destroyAllWindows()
    
    # 删除所有生成的json文件
    delete_image()

    sys.exit(0)

if __name__ == '__main__':
    w = Watcher()
    w.run()
