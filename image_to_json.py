import json
import shutil
import signal
import sys
import time
import os
import cv2
import mediapipe as mp
from pathlib import Path
from ultralytics import YOLO
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 处理SIGTERM信号
signal.signal(signal.SIGTERM, quit)
# 处理SIGINT信号
signal.signal(signal.SIGINT, quit)

delete_time = time.time()
last_saved_time = time.time()

class LandmarksHandler(FileSystemEventHandler):
    def __init__(self, output_folder):
        self.output_folder = output_folder

        # 初始化YOLO模型
        model_path = Path("./yolov8m.pt")
        self.model = YOLO(model_path)

        # 选择模型包（0：轻量模型，1：默认模型，2：全身模型）
        model_complexity = 0

        # 初始化姿势检测器
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                    model_complexity=model_complexity,
                                    min_detection_confidence=0.6,
                                    min_tracking_confidence=0.6)

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.jpg'):

            # 防止出现文件读写冲突或未完全写入问题
            time.sleep(0.3)
            self.process_image(event.src_path)

    def process_image(self, image_path):
        try:
            image_path = Path(image_path)

            if not image_path.exists():
                print(f"File {image_path} does not exist.")
                return
            
            image = cv2.imread(str(image_path))
            json_filename = image_path.stem

            print(f"Processing image {image_path}")

            # 使用YOLOv8检测人物,提高YOLOv8的检测置信度
            # results = self.model(source=image, show=False, conf=0.4)
            results_tr = self.model.track(source=image, conf=0.4, persist=True)
            print("model run")

            # 只保留检测到的人的结果
            # person_results = []
            # for result in results:
            #     for box in result.boxes:
            #         if box.cls == 0:  # 类别0表示“person”
            #             person_results.append(box)

            # 存储所有人的关键点数据
            landmarks_data = []  
            # 下标提取
            indices = [0, 2, 5, 11, 12, 
                       13, 14, 15, 16, 17, 
                       18, 19, 20, 21, 22, 
                       23, 24, 25, 26, 27, 
                       28, 29, 30, 31, 32]
            # 骨骼关键点对应的键
            keys = [
                "nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder", 
                "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky", 
                "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
                "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", 
                "right_ankle", "left_heel", "right_hell", "left_foot_index", "right_foot_index"
            ]

            # 初始化 goal_data
            goal_data = {"users": [], "protoName": "PoseMsg"}
            # 处理跟踪结果
            tracked_objects = results_tr[0].boxes
            for obj in tracked_objects:
                if int(obj.cls[0]) != 0:  # 类别0表示“person”
                    continue
                x1, y1, x2, y2 = map(int, obj.xyxy[0])  # 获取边界框坐标
                # person_roi = image[y1:y2, x1:x2]  # 裁剪出人物区域
                track_id = int(obj.id[0]) if obj.id is not None else 'N/A'  # 获取跟踪ID
                # 裁剪出人物区域
                person_roi = image[y1:y2, x1:x2]
                
                # 将裁剪出的区域转换为RGB，因为MediaPipe使用的是RGB图像
                image_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                result_pose = self.pose.process(image_rgb)  # 使用MediaPipe进行姿势识别
                
                if result_pose.pose_world_landmarks:
                    dic = {}
                    
                    for i, key in enumerate(keys):
                        idx = indices[i]
                        landmark = result_pose.pose_world_landmarks.landmark[idx]
                        # cx = int(landmark.x * (x2 - x1)) + x1
                        # cy = int(landmark.y * (y2 - y1)) + y1
                        # cz = landmark.z
                        dic[key] = {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        }
                        print(landmark.visibility)
                        
                    landmarks_data.append({"id": str(len(landmarks_data)), "position": dic})
                    self.mp_drawing.draw_landmarks(
                        image[y1:y2, x1:x2], result_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    # 绘制跟踪ID
                    cv2.putText(image, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    
                goal_data = {"users": landmarks_data}
                # goal_data["protoName"] = "JsonSend"

                # 将关键点重新映射回原图
                # for landmark in person_landmarks['landmarks']:
                #     cv2.circle(image, (landmark['x'], landmark['y']), 5, (0, 255, 0), -1)

                # 绘制检测到的姿势
                self.mp_drawing.draw_landmarks(
                    image[y1:y2, x1:x2], result_pose.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # 保存关键点数据
            save_json(goal_data, json_filename)

            # cv2.imshow('MediaPipe Pose', image)
            # 按下ESC键退出
            if cv2.waitKey(5) & 0xFF == 27:
                return

        except PermissionError as e:
            print(f"PermissionError: {e}. Retrying in 1 second...")
            time.sleep(1)
            self.process_image(image_path)


def save_json(landmarks_data, json_filename):
    global last_saved_time
    current_time = time.time()
    if current_time - last_saved_time >= 0.5:
        last_saved_time = current_time

        filename = f"{output_folder}/{json_filename}.json"
        # 保存关键点数据
        with open(filename, 'w') as f:
            json.dump(landmarks_data, f, indent=4)
        print(f"Saved landmarks data to {filename}")
    delete_folders(folders_to_clean)


def monitor_folder(input_folder, output_folder):
    event_handler = LandmarksHandler(output_folder)
    observer = Observer()
    observer.schedule(event_handler, path=input_folder, recursive=False)
    observer.start()
    print(f"Monitoring {input_folder} for new JPG files...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def delete_folders(folders, max_files=720, delete_count=700):
    global delete_time
    current_time = time.time()
    # 经过一段时间后定时清理超过数量的文件
    if current_time - delete_time > 10:    
        delete_time = current_time
        for folder in folders:
            files = sorted(os.listdir(folder), key=lambda x: os.path.getmtime(os.path.join(folder, x)))
            if len(files) > max_files:
                files_to_delete = files[:delete_count]
                for filename in files_to_delete:
                    file_path = os.path.join(folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')


def delete_all_folders(folders):
    for folder in folders:
        files = sorted(os.listdir(folder), key=lambda x: os.path.getmtime(os.path.join(folder, x)))
        max_files = len(files)
        files_to_delete = files[:max_files]
        for filename in files_to_delete:
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')


def quit():
    delete_all_folders(folders_to_clean)
    print("Exiting...")
    sys.exit(0)


if __name__ == "__main__":
    # 相关路径
    # input_folder = './jpg_path/' 
    input_folder = "D:\Code\Python\jpg_path"
    # output_folder = './json_save/' 
    output_folder = "D:\Code\Python\json_save"
    folders_to_clean = [input_folder, output_folder]
    delete_all_folders(folders_to_clean)

    monitor_thread = Thread(target=monitor_folder, args=(input_folder, output_folder))
    monitor_thread.start()

    try:
        monitor_thread.join()  # 等待线程完成
    except KeyboardInterrupt:
        quit()