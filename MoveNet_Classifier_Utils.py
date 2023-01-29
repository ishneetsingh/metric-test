import cv2
import tensorflow as tf
import numpy as np
import time

RECTANGLE_COLOURS = {
    0: (204, 225, 242),
    1: (251, 247, 213),
    2: (245, 205, 222)
}

POSE_NAMES = [
    "Standing",
    "Sitting",
    "Lying"
]

def load_classifier():
    return tf.lite.Interpreter(model_path='./model/model.tflite')

CLASSIFIER = load_classifier()
CLASSIFIER.allocate_tensors()

def classifier_prediction_for_person(keypoints_of_person, frame, conf_threshold, coords, x_box, y_box,  n_features=51):
    model_time_start = time.time()

    y = int(0.5 * (coords[15] + coords[18]))
    x = int(0.5 * (coords[16] + coords[19]))
    box_coords = [y, x]

    temp = np.reshape(keypoints_of_person, (1, n_features, 1))

    # Setup input and output (Classifier)
    classifier_in = CLASSIFIER.get_input_details()
    classifier_out = CLASSIFIER.get_output_details()

    # Make Classifications
    CLASSIFIER.set_tensor(classifier_in[0]['index'], np.array(temp))
    CLASSIFIER.invoke()
    results = CLASSIFIER.get_tensor(classifier_out[0]['index'])[0]

    classified_pose = np.argmax(results)

    model_time_end = time.time()
    model_time = model_time_end - model_time_start

    prob = f"{round(max(results)*100, 2)}%"

    if max(results) > conf_threshold:
        draw_time_1_start = time.time()
        draw_classifying_box(frame, box_coords, classified_pose, prob, x_box, y_box)
        draw_time_1_end = time.time()

        draw_time_1 = draw_time_1_end - draw_time_1_start

    return model_time, draw_time_1



def draw_classifying_box(frame, coords, classified_pose, prob, x_box, y_box):
    y, x = coords
    thickness = int(x_box / 25)
    scale = x_box / 85

    if thickness == 0:
        thickness = 1

    # Draw box
    cv2.rectangle(frame,
        (x - int(x_box), y - int(y_box)),
        (x + int(x_box), y + int(y_box)),
        tuple(RECTANGLE_COLOURS[classified_pose]), -1)

    # Classified pose
    cv2.putText(frame, POSE_NAMES[classified_pose], [x - int(14*x_box/15), y - int(y_box/3)],
        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Probabilities
    cv2.putText(frame, prob, [x - int(14*x_box/15), y + int(2*y_box/3)],
        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)


