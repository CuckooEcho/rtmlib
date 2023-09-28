import time

import cv2

from rtmlib import YOLOX, RTMPose, draw_bbox, draw_skeleton

# import numpy as np

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

# img = cv2.imread('./demo.jpg')
cap = cv2.VideoCapture('./demo.jpg')

openpose_skeleton = False  # True for openpose-style, False for mmpose-style

det_model = YOLOX('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',  # noqa
                  model_input_size=(640, 640),
                  backend=backend,
                  device=device)
pose_model = RTMPose('https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip',  # noqa
                     model_input_size=(288, 384),
                     to_openpose=openpose_skeleton,
                     backend=backend,
                     device=device)

video_writer = None
pred_instances_list = []
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()
    bboxes = det_model(frame)
    det_time = time.time() - s
    print('det: ', det_time)

    img_show = frame.copy()
    img_show = draw_bbox(img_show, bboxes)

    s = time.time()
    keypoints, scores = pose_model(frame, bboxes=bboxes)
    pose_time = time.time() - s
    print('pose: ', pose_time)

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.5)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey()
