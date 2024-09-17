import cv2
import mediapipe as mp
import numpy as np
import time

from ultralytics import YOLO

frame_count = 0 #初始化帧数计数器

model = YOLO("./yolov8m.pt")


#初始化姿势模块
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = False,
                    min_detection_confidence = 0.6,
                    min_tracking_confidence = 0.6)
mp_draw = mp.solutions.drawing_utils

#打开摄像头
cap = cv2.VideoCapture(".\Yolov8\1.mp4")

#跟踪人物的字典
people_tracker = {}
next_person_id = 0

while cap.isOpened():
    #读取摄像头图像
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue



    # 使用YOLOv8检测人物,提高YOLOv8的检测置信度
    results = model(source=image, show=True, conf=0.4)

    # 只保留检测到的人的结果
    person_results = []
    for result in results:
        for box in result.boxes:
            if box.cls == 0:  # 类别0表示“person”
                person_results.append(box)

    for box in person_results:
        # 将张量转换为NumPy数组并展平
        # x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy().flatten())  

        # 提取人物区域
        # person_image = image[y1:y2, x1:x2]
        # person_image = image

        # # 将图像从BGR格式转换为RGB格式
        # image_rgb = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
        person_image = image
        #转换图像 BGR转RGB
        image2 = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
        result_pose = pose.process(image2)

        # 保存每个人的关键点数据 TODO:改变json格式
        # if result_pose.pose_world_landmarks:
        #     person_landmarks = {
        #         'id': len(landmarks_data),  # 使用列表的长度作为ID
        #         'landmarks': []
        #     }
        #     for landmark in result_pose.pose_world_landmarks.landmark:
        #         cx = int(landmark.x * (x2 - x1)) + x1
        #         cy = int(landmark.y * (y2 - y1)) + y1
        #         cz = landmark.z
        #         person_landmarks['landmarks'].append({
        #             'x': cx,
        #             'y': cy,
        #             'z': cz
        #         })
        #     landmarks_data.append(person_landmarks)
        
        if result_pose.pose_world_landmarks:
            mp_draw.draw_landmarks(
                # image[y1:y2, x1:x2], result_pose.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS)
                image, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            #显示图像
            cv2.imshow('MediaPipe Pose', image)
    #处理图像
    # results = pose.process(image)

    # 每处理30帧，输出一次pose_landmarks数据
    # frame_count += 1
    # if frame_count % 30 == 0:
    #     frame_count = 0
    #     print(results.pose_world_landmarks)

    #转换图像 RGB转BGR
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # #记录当前帧的人物特征
    # current_frame_features = []

    # #绘制姿势关键点
    # if results.pose_landmarks:
    #     # 初始化边界框的最大和最小坐标
    #     min_x, min_y = float('inf'), float('inf')
    #     max_x, max_y = 0, 0    

    #     for landmark in results.pose_landmarks.landmark:
    #         #提取特征
    #         feature = (landmark.x, landmark.y)
    #         current_frame_features.append(feature)

    #         # 更新边界框坐标
    #         min_x, min_y = min(min_x, landmark.x), min(min_y, landmark.y)
    #         max_x, max_y = max(max_x, landmark.x), max(max_y, landmark.y)


    #     # 匹配和跟踪逻辑
    #     # 这里简化处理，实际应用需要更复杂的匹配算法
    #     for feature in current_frame_features:
    #         found_match = False
    #         for person_id, person_feature in people_tracker.items():
    #             some_threshold = 0.5  # 设置阈值
    #             if np.linalg.norm(np.array(feature) - np.array(person_feature)) < some_threshold:
    #                 found_match = True
    #                 people_tracker[person_id] = feature  # 更新特征
    #                 break
    #         if not found_match:
    #             people_tracker[next_person_id] = feature
    #             next_person_id += 1

    #     # 绘制边界框
    #     # 需要将归一化坐标转换为像素坐标
    #     height, width, _ = image.shape
    #     start_point = (int(min_x * width), int(min_y * height))
    #     end_point = (int(max_x * width), int(max_y * height))
    #     color = (255, 0, 0)  # BGR
    #     thickness = 2
    #     cv2.rectangle(image, start_point, end_point, color, thickness)

        
    #     mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    #显示图像
    # cv2.imshow('MediaPipe Pose', image)

    #按下ESC键退出
    if cv2.waitKey(5) & 0xFF == 27:
        break

#释放资源
cap.release()
cv2.destroyAllWindows()  