# import signal
# import sys

# def custom_exit(signum, frame):
#     with open('./abcd.txt', 'w') as file:
#         file.write('abcd')
#     print(f"Exiting script due to signal {signum}")
#     sys.exit(0)

# # 处理SIGTERM信号
# signal.signal(signal.SIGTERM, custom_exit)
# # 处理SIGINT信号
# signal.signal(signal.SIGINT, custom_exit)

# print("Script is running. Send SIGTERM or SIGINT to exit.")

# try:
#     while True:
#         # 模拟脚本的主要逻辑
#         pass
# except KeyboardInterrupt:
#     print("\nKeyboard interrupt detected. Exiting script...")
#     custom_exit(None, None)


import json
import os
import cv2
from ultralytics import YOLO

model = YOLO("./yolov8m-pose.pt")
print(model)
# video_path = 'Yolov8/1.mp4'
# video_path = "C:/Users/Liu/Desktop/100083.jpg"
video_path = "./input/100083.jpg"
cap = cv2.VideoCapture(video_path)
time_count = 0

# 获取文件名
file_name = os.path.basename(video_path)

# 存储字典形式的关键点数据
keypoints_dict = {}

while cap.isOpened():
    # 读取摄像头图像
    success, image = cap.read()
    if not success:
        print("忽略空视频帧")
        time_count += 1
        if time_count > 20:
            break
        continue

    img_height, img_width = image.shape[:2]

    # 存储每个人的关键点数据
    person_keypoints_list = []    

    # Run inference
    # results = model(image, show=True, save_dir='output', conf=0.6)
    results = model(image, conf=0.6)

        # 对每一帧结果进行处理
    for result in results:
        for person in result.keypoints:
            frame_keypoints = []
            # 提取关键点的x, y坐标和置信度
            keypoints_xy = person.xy  # 提取关键点坐标
            keypoints_conf = person.conf  # 提取关键点置信度

            # 遍历每个关键点，组合 [x, y, confidence]
            for kp, conf in zip(keypoints_xy[0], keypoints_conf[0]):
                x, y = kp.tolist()  # 将tensor转化为list
                x_normalized = (x / img_width) * 2 - 1  # 将x归一化为[-1, 1]
                y_normalized = (y / img_height) * 2 - 1  # 将y归一化为[-1, 1]
                frame_keypoints.append([x, y, conf.item(), x_normalized, y_normalized])
                # frame_keypoints.append([x, y, conf.item()])  # 组合 [x, y, confidence]

            # 将这一帧的关键点存储到结果列表中
            person_keypoints_list.append(frame_keypoints)

        # 将此帧的所有人物关键点加入字典，使用文件名作为键

    if person_keypoints_list:
        keypoints_dict[file_name] = person_keypoints_list

    # 由于只是处理图片，可以在这里直接退出循环
    break

# 保存关键点数据为 JSON 文件
print(keypoints_dict)
output_json_path = 'output_keypoints.json'
with open(output_json_path, 'w') as f:
    f.write('{\n')
    key_count = len(keypoints_dict)
    for i, (key, value) in enumerate(keypoints_dict.items()):
        f.write(f'  "{key}": [\n')
        person_count = len(value)
        for j, person in enumerate(value):
            f.write('    [\n')
            keypoint_count = len(person)
            for k, keypoint in enumerate(person):
                if k < keypoint_count - 1:
                    f.write(f'      {keypoint},\n')  # 每个关键点占一行
                else:
                    f.write(f'      {keypoint}\n')  # 最后一个关键点不加逗号
            if j < person_count - 1:
                f.write('    ],\n')
            else:
                f.write('    ]\n')  # 最后一个人不加逗号
        if i < key_count - 1:
            f.write('  ],\n')
        else:
            f.write('  ]\n')  # 最后一个键值对不加逗号
    f.write('}\n')
# with open(output_json_path, 'w') as f:
#     f.write('{\n')
#     for key, value in keypoints_dict.items():
#         f.write(f'  "{key}": [\n')
#         for person in value:
#             f.write('    [\n')
#             for keypoint in person:
#                 f.write(f'      {keypoint},\n')  # 每个关键点占一行
#             f.write('    ],\n')
#         f.write('  ],\n')
#     f.write('}\n')
    # json.dump(keypoints_dict, f, separators=(',', ':'), ensure_ascii=False)
    # json.dump(keypoints_dict, f, indent=4)

print(f"关键点数据已保存为 {output_json_path}")

cap.release()
cv2.destroyAllWindows()

    # for result in results:
    #     print(result.keypoints.xy, result.keypoints.conf)
    #     for person in result.keypoints:  # 遍历检测到的每个人

    #         keypoints = person.xy
    #         keypoints_list = []

    #         for (x, y) in keypoints:                
    #             # 将归一化坐标转换为像素坐标
    #             pixel_x = int(x * img_width)
    #             pixel_y = int(y * img_height)
    #             keypoints_list.append([pixel_x, pixel_y, 0, 0, 0])

    #         keypoints_data.append(keypoints_list)

    # # 保存关键点数据到 JSON 文件
    # output_file = 'keypoints_output.json'
    # with open(output_file, 'w') as f:
    #     json.dump(keypoints_data, f, indent=4)

    # print(f'关键点坐标已保存到 {output_file}')


# import json
# import os

# # 确保输出目录存在
# output_dir = 'output'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # 对图片进行处理的循环
# for img_name in sorted(pose2d_result.keys()):
#     img_path = osp.join(img_dir, img_name)

#     original_img = cv2.imread(img_path)
#     if original_img is None:
#         print(f"Failed to load image: {img_path}")
#         continue

#     input2 = original_img.copy()
#     original_img_height, original_img_width = original_img.shape[:2]
#     coco_joint_list = pose2d_result[img_name]

#     if args.img_idx not in img_name:
#         continue

#     drawn_joints = []
#     c = coco_joint_list

#     for idx in range(len(coco_joint_list)):
#         pose_thr = 0.1
#         coco_joint_img = np.asarray(coco_joint_list[idx])[:, :3]
#         coco_joint_img = add_pelvis(coco_joint_img, coco_joints_name)
#         coco_joint_img = add_neck(coco_joint_img, coco_joints_name)
#         coco_joint_valid = (coco_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)

#         det_score = sum(coco_joint_img[:, 2])
#         if det_score < 1.0:
#             continue

#         tmp_joint_img = coco_joint_img.copy()
#         continue_check = False
#         for ddx in range(len(drawn_joints)):
#             drawn_joint_img = drawn_joints[ddx]
#             drawn_joint_val = (drawn_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
#             diff = np.abs(tmp_joint_img[:, :2] - drawn_joint_img[:, :2]) * coco_joint_valid * drawn_joint_val
#             diff = diff[diff != 0]
#             if diff.size == 0:
#                 continue_check = True
#             elif diff.mean() < 20:
#                 continue_check = True 
#         if continue_check:
#             continue
#         drawn_joints.append(tmp_joint_img)

#         bbox = get_bbox(coco_joint_img, coco_joint_valid[:, 0])  # xmin, ymin, width, height
#         bbox = process_bbox(bbox, original_img_width, original_img_height)
#         if bbox is None:
#             continue
#         img, img2bb_trans, bb2img_trans = generate_patch_image(input2[:,:,::-1], bbox, 1.0, 0.0, False, cfg.input_img_shape)
#         img = transform(img.astype(np.float32)) / 255
#         img = img.cuda()[None,:,:,:]

#         coco_joint_img_xy1 = np.concatenate((coco_joint_img[:, :2], np.ones_like(coco_joint_img[:, :1])), 1)
#         coco_joint_img[:, :2] = np.dot(img2bb_trans, coco_joint_img_xy1.transpose(1, 0)).transpose(1, 0)
#         coco_joint_img[:, 0] = coco_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
#         coco_joint_img[:, 1] = coco_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

#         coco_joint_img = transform_joint_to_other_db(coco_joint_img, coco_joints_name, joints_name)
#         coco_joint_valid = transform_joint_to_other_db(coco_joint_valid, coco_joints_name, joints_name)
#         coco_joint_valid[coco_joint_img[:, 2] <= pose_thr] = 0

#         coco_joint_trunc = coco_joint_valid * ((coco_joint_img[:, 0] >= 0) * (coco_joint_img[:, 0] < cfg.output_hm_shape[2]) * (coco_joint_img[:, 1] >= 0) * (coco_joint_img[:, 1] < cfg.output_hm_shape[1])).reshape(
#             -1, 1).astype(np.float32)
#         coco_joint_img, coco_joint_trunc, bbox = torch.from_numpy(coco_joint_img).cuda()[None, :, :], torch.from_numpy(coco_joint_trunc).cuda()[None, :, :], torch.from_numpy(bbox).cuda()[None, :]

#         inputs = {'img': img, 'joints': coco_joint_img, 'joints_mask': coco_joint_trunc}
#         targets = {}
#         meta_info = {'bbox': bbox}

#         with torch.no_grad():
#             out = model(inputs, targets, meta_info, 'test')

#         # 获取并保存3D坐标
#         mesh_cam_render = out['mesh_cam_render'][0].cpu().numpy()

#         output_data = {"3D_coordinates": mesh_cam_render.tolist()}
#         with open(f'{output_dir}/3D_coordinates_{img_name}.json', 'w') as f:
#             json.dump(output_data, f)

# print('3D坐标已保存到JSON文件中')
