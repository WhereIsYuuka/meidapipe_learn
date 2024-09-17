import cv2
import mediapipe as mp

#初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False,
                       max_num_hands = 2,
                       min_detection_confidence = 0.5,
                       min_tracking_confidence = 0.5)
mp_draw = mp.solutions.drawing_utils

#打开摄像头
#解释：cap = cv2.VideoCapture(0)表示打开默认摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    #读取摄像头图像 success表示是否成功，image表示图像
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    #转换图像 BGR转RGB，很多算法和模型都期望输入图像是RGB格式的
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #处理图像
    results = hands.process(image)

    #转换图像 RGB转BGR，大多数图像处理库和显示设备都使用BGR格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #绘制手部关键点
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #这个函数接受三个参数：图像、手部关键点的坐标数据和手部关键点之间的连接关系
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('MediaPipe Hands', image)
    '''cv2.waitKey()函数是一个用于等待键盘输入的函数，
    cv2.waitKey()函数的返回值是按下的键的ASCII码值。
    在这个例子中，代码使用了位运算符&和0xFF来提取返回值的最低8位，
    然后与27进行比较。如果按下的键的ASCII码值等于27（即按下了ESC键），则会跳出循环。'''
    if cv2.waitKey(5) & 0xFF == 27:
        break

#释放资源
cap.release()
cv2.destroyAllWindows()