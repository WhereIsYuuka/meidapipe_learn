import glob
import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import json
import time
import numpy as np

# 初始化模型
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
model = YOLO("./yolov8m.pt")  # 请确保你有预训练的YOLOv8模型权重文件

# 初始化视频捕获
cap = cv2.VideoCapture(0)  # 捕获摄像头视频

last_saved_time = time.time()

# 保存关键点数据的函数
def save_landmarks(landmarks, filename):
    with open(filename, 'w') as f:
        json.dump(landmarks, f)

def main_run():
# 初始化Mediapipe Pose
    try:
        with mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # TODO: 检测文件夹

                # 使用YOLOv8检测人物
                results = model(frame, show=False, conf=0.4)  # 提高YOLOv8的检测置信度
                
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
                    person_image = frame[y1:y2, x1:x2]

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
                            cv2.circle(frame, (landmark['x'], landmark['y']), 5, (0, 255, 0), -1)

                        # 绘制检测到的姿势
                        mp_drawing.draw_landmarks(
                            frame[y1:y2, x1:x2], result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                current_time = time.time()
                if current_time - last_saved_time >= 0.5:  # 每半秒记录一次
                    last_saved_time = current_time
                    timestamp = int(round(current_time * 1000))
                    filename = f"./output/landmarks_{timestamp}.json"
                    save_landmarks(landmarks_data, filename)
                    print(f"Saved landmarks data to {filename}")

                cv2.imshow('Mediapipe & YOLOv8', frame)

                if cv2.waitKey(5) & 0xFF == 27:
                    break

    finally:  # 无论如何都会执行的代码块
        cap.release()
        cv2.destroyAllWindows()
        
        # 删除所有生成的json文件
        for filename in glob.glob("./output/landmarks_*.json"):
            os.remove(filename)
        print("All generated JSON files have been deleted.")

if __name__ == '__main__':
    main_run()