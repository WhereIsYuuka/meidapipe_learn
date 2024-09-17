import cv2
import mediapipe as mp
from ultralytics import YOLO
import json
import time
import numpy as np

# 初始化模型
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
model = YOLO("yolov8m.pt")  # 请确保你有预训练的YOLOv8模型权重文件
pTime = 0

# 初始化视频捕获
cap = cv2.VideoCapture(0)  # 捕获摄像头视频

# 保存关键点数据的函数
def save_landmarks(landmarks, filename):
    with open(filename, 'w') as f:
        json.dump(landmarks, f)

last_saved_time = time.time()
next_person_id = 0  # 初始化人物ID

# 用于存储前一帧的人物位置和ID
previous_people = []

# 初始化Mediapipe Pose
with mp_pose.Pose(static_image_mode=False,
                  min_detection_confidence=0.6,
                  min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 使用YOLOv8检测人物，过滤掉非人物类别
        results = model(frame, show=False, conf=0.4)
        
        # 只保留检测到的人的结果
        person_results = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # 类别0表示“person”
                    person_results.append(box)

        current_people = []  # 当前帧中的人物位置和ID
        landmarks_data = []  # 存储所有人的关键点数据

        for box in person_results:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())  # 将张量转换为NumPy数组并展平
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 提取人物区域
            person_image = frame[y1:y2, x1:x2]

            # 将图像从BGR格式转换为RGB格式
            image_rgb = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)

            # 进行姿势检测
            result_pose = pose.process(image_rgb)

            # 确定每个人的ID
            person_id = None
            min_distance = float('inf')
            for prev_person in previous_people:
                prev_id, prev_center_x, prev_center_y = prev_person
                distance = np.sqrt((center_x - prev_center_x)**2 + (center_y - prev_center_y)**2)
                if distance < min_distance and distance < 50:  # 距离阈值
                    min_distance = distance
                    person_id = prev_id

            if person_id is None:
                person_id = next_person_id
                next_person_id += 1

            current_people.append((person_id, center_x, center_y))

            # 保存每个人的关键点数据
            if result_pose.pose_landmarks:
                person_landmarks = {
                    'id': person_id,
                    'landmarks': []
                }
                for landmark in result_pose.pose_landmarks.landmark:
                    cx, cy = int(landmark.x * (x2 - x1)), int(landmark.y * (y2 - y1))
                    cz = landmark.z
                    person_landmarks['landmarks'].append({
                        'x': cx + x1,
                        'y': cy + y1,
                        'z': cz
                    })
                landmarks_data.append(person_landmarks)

                # 将关键点重新映射回原图
                for landmark in person_landmarks['landmarks']:
                    cv2.circle(frame, (landmark['x'], landmark['y']), 5, (0, 255, 0), -1)

            # 绘制检测到的姿势
            if result_pose.pose_landmarks:
                mp_drawing.draw_landmarks(
                    person_image, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cTime = time.time() #处理完一帧图像的时间
            fps = 1/(cTime-pTime)#即为FPS
            pTime = cTime  #重置起始时间

            # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
            cv2.putText(frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  

        # 更新前一帧的人物位置和ID
        previous_people = current_people

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

cap.release()
cv2.destroyAllWindows()
