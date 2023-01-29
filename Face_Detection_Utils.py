import numpy as np
import cv2
import time

def face_bbox(coords):
    face_box_time_start = time.time()

    keypoints_to_consider = [0, 1, 2, 3, 4, 5, 6, 11, 12]
    y_coords = [coords[::3][idx] for idx in keypoints_to_consider]
    x_coords = [coords[1::3][idx] for idx in keypoints_to_consider]

    x_body_range = max(x_coords) - min(x_coords)
    y_body_range = max(y_coords) - min(y_coords)
    ratio = x_body_range / y_body_range

    x_face = x_coords[:5]
    y_face = y_coords[:5]
    x_bar = np.mean(x_coords[:5])
    y_bar = np.mean(y_coords[:5])

    # Get largest |x_i - x_bar|
    x_max = -1
    for x_i in x_face:
        temp = abs(x_i - x_bar)
        if temp > x_max:
            x_max = temp

    # Get largest |y_i - y_bar|
    y_max = -1
    for y_i in y_face:
        temp = abs(y_i - y_bar)
        if temp > y_max:
            y_max = temp

    if ratio > 1:
        h_head = 2.5 * y_max
        w_head = 1.5 * h_head
    else:
        w_head = 2.5 * x_max
        h_head = 1.5 * w_head

    face_box_time_end = time.time()
    face_box_time = face_box_time_end - face_box_time_start

    return (int(x_bar - 0.5*w_head), int(y_bar - 0.5*h_head), int(x_bar + 0.5*w_head), int(y_bar + 0.5*h_head), face_box_time)

def blur_face(frame, coords):
    face_time_start = time.time()

    height, width = frame.shape[:2]
    kernel_size = int(0.0000076853 * height * width + 10.0335)
    if kernel_size % 2 == 0:
        kernel_size += 1

    x1, y1, x2, y2, face_box_time = face_bbox(coords)

    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0
    if x1 > width:
        x1 = width
    if x2 > width:
        x2 = width
    if y1 > height:
        y1 = height
    if y2 > height:
        y2 = height

    try:
        range_to_blur = frame[y1:y2, x1:x2]
        range_to_blur = cv2.GaussianBlur(range_to_blur, (kernel_size, kernel_size), 0)
        frame[y1:y2, x1:x2] = range_to_blur
    except:
        return -1, -1

    face_time_end = time.time()

    blurring_time = face_time_end - face_time_start - face_box_time
    return face_box_time, blurring_time