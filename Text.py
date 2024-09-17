import socket
import os
import mediapipe as mp
import cv2
#有时候反复启动端口没有释放，需要杀死数据端口进程
def kill_port(port):
    print("try to kill %s pid..." % port)
    find_port= 'netstat -aon | findstr %s' % port
    result = os.popen(find_port)
    text = result.read()
    pid= text[-5:-1]
    if pid == "":
        print("not found %s pid..." % port)
        return
    else:
        find_kill= 'taskkill -f -pid %s' % pid
        result = os.popen(find_kill)
        return result.read()

def camera():
    cap = cv2.VideoCapture(0)

    #这里用的是Mediapipe的Holistic模型
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1,
                              smooth_landmarks=True) as holistic:
        while True:
            ret, image = cap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pos1 = pos2 = poseString = ""
            if results.left_hand_landmarks:
                # 可以选择画出左手识别结果
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                for lm in results.left_hand_landmarks.landmark:
                    pos1 += f'{lm.x},{lm.y},{lm.z},'
                pos1 += 'Left,'
            if results.right_hand_landmarks:
                # 可以选择画出右手识别结果
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                for lm in results.right_hand_landmarks.landmark:
                    pos2 += f'{lm.x},{lm.y},{lm.z},'
                pos2 += 'Right,'
            if results.pose_landmarks:
                # 可以选择画出全身识别结果
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                for lm in results.pose_landmarks.landmark:
                    poseString += f'{lm.x},{lm.y},{lm.z},'

            # 手部数据
            date1 = pos1 + pos2 + ';'
            # 身体数据
            date2 = poseString
            # 使用UDP发送识别数据给Unity端
            sock.sendto(str.encode(date1 + date2), serverAddressPort)
            cv2.imshow('0', image)
            if cv2.waitKey(10) == 27:
                break


if __name__ == '__main__':
    kill_port(5054)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 向5054端口发送数据
    serverAddressPort = ('127.0.0.1', 5054)
    camera()