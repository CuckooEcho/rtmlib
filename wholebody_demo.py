import time

import cv2

from rtmlib import PoseTracker, Wholebody, draw_skeleton

import numpy as np

device = 'cuda'  # cpu, cuda, mps
backend = 'onnxruntime'  # opencv, onnxruntime, openvino

video_file = 'def.mp4'


cap = cv2.VideoCapture(video_file)

# 定义保存视频的相关参数
output_video_path = f'rtmlib_{video_file}'  # 输出视频的路径，可根据需求修改
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器，这里使用mp4v，可根据需求调整为其他合适的
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原视频的帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取原视频宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取原视频高度
out = cv2.VideoWriter(output_video_path, fourcc, fps,
                      (width, height))  # 创建VideoWriter对象用于保存视频

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

wholebody = PoseTracker(
    Wholebody,
    det_frequency=7,
    to_openpose=openpose_skeleton,
    mode='performance',  # balanced, performance, lightweight
    backend=backend,
    device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()
    keypoints, scores = wholebody(frame)
    det_time = time.time() - s
    print('det: ', det_time)

    img_show = frame.copy()

    h, w = img_show.shape[:2]

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)
    # print(scores)
    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=4)

    img_show = cv2.resize(img_show, (w, h))
    cv2.imshow('img', img_show)
    out.write(img_show)
    cv2.waitKey(10)
# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
