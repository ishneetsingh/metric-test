import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from MoveNet_Drawing_Utils import draw_skeleton
from MoveNet_Classifier_Utils import classifier_prediction_for_person
from Face_Detection_Utils import blur_face
from sklearn.preprocessing import StandardScaler
import time

# StandardScaler mean variance and scale attributes based on Training CSV file
MEAN = np.array([0.41854253, 0.48043439, 0.46034951, 0.41260677, 0.48326657, 0.4652356,
                0.41315069, 0.47737307, 0.46812437, 0.4140444,  0.48688148, 0.45220497,
                0.41490925, 0.47128035, 0.45992398, 0.43869948, 0.4910098,  0.47121129,
                0.44072646, 0.46718979, 0.48494591, 0.48209117, 0.50375735, 0.35644908,
                0.48666772, 0.46572357, 0.35427947, 0.4967549,  0.50433978, 0.3298676,
                0.5031243,  0.47378601, 0.3366836,  0.51254528, 0.4928401,  0.44121969,
                0.51430082, 0.47654498, 0.4574543,  0.5470065,  0.50484993, 0.34879261,
                0.54840444, 0.48924357, 0.34695526, 0.59995424, 0.5064794,  0.35923976,
                0.59969802, 0.49193385, 0.35454279])
VARIANCE = np.array([0.00755973, 0.0090774,  0.01884628, 0.00799929, 0.00970984, 0.01826976,
                    0.00801717, 0.00941241, 0.01851853, 0.00787304, 0.00998578, 0.01858428,
                    0.00788959, 0.00937987, 0.01700437, 0.00504892, 0.00906765, 0.01949288,
                    0.00515518, 0.00777827, 0.01958251, 0.00416729, 0.00882446, 0.01757093,
                    0.00457245, 0.00734585, 0.01643724, 0.00463145, 0.00950063, 0.01546841,
                    0.00509391, 0.00832346, 0.01422947, 0.0023061,  0.00509067, 0.01854865,
                    0.00247501, 0.00438542, 0.01839134, 0.00293952, 0.00589606, 0.01710563,
                    0.00307021, 0.00528049, 0.0152193,  0.00644599, 0.00891035, 0.02130368,
                    0.0067696,  0.00890602, 0.01812211])
SCALE = np.array([0.08694671, 0.09527538, 0.13728175, 0.08943875, 0.09853853, 0.13516567,
                0.08953863, 0.09701756, 0.13608282, 0.08873016, 0.09992887, 0.13632417,
                0.08882339, 0.09684972, 0.13040079, 0.07105578, 0.09522422, 0.13961691,
                0.07179961, 0.08819452, 0.13993752, 0.06455452, 0.09393862, 0.13255539,
                0.06761991, 0.08570794, 0.12820781, 0.06805476, 0.0974712,  0.12437205,
                0.0713716,  0.09123303, 0.11928735, 0.04802191, 0.07134893, 0.13619344,
                0.04974945, 0.06622251, 0.13561469, 0.05421731, 0.07678583, 0.1307885,
                0.05540947, 0.07266699, 0.12336654, 0.08028692, 0.09439466, 0.14595782,
                0.08227755, 0.09437171, 0.1346184])
scaler = StandardScaler()
scaler.mean_ = MEAN
scaler.var_ = VARIANCE
scaler.scale_ = SCALE

def load_movenet():
    interpreter = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    return interpreter.signatures['serving_default']
MOVENET = load_movenet()

# Significant Features
IDX_TO_KEEP = [42,50, 39, 29, 32, 30, 27, 37, 25, 19, 31, 36, 33, 13, 43,  7,  1, 34,  4, 10, 16, 49, 28, 22, 40, 46]

def get_affine_transform_to_fixed_sizes_with_padding(size, new_sizes):
    width, height = new_sizes
    scale = min(height / float(size[1]), width / float(size[0]))
    M = np.float32([[scale, 0, 0], [0, scale, 0]])
    M[0][2] = (width - scale * size[0]) / 2
    M[1][2] = (height - scale * size[1]) / 2
    return M


def movenet_processing(frame, max_people=1, mn_conf=0.2, kp_conf=0, pred_conf=0, draw_movenet_skeleton = True, blur_faces = True):
    '''
    Runs MoveNet and Pose Classifier on frame, returns frame with MoveNet skeleton and pose classifying box.

    This function will also return (in seconds):
    - Time taken to draw classified pose + face blurring onto frame
    - Time taken to classify pose (MN + classifier)
    - Time taken to detect faces (MN + calculation)

    Args:
        - frame: NumPy array of frame (in RGB format)
        - max_people: Maximum number of people to be detected by MoveNet
        - mn_conf: Minimum average confidence of MoveNet keypoints for pose classifier to be run
        - kp_conf: Minimum confidence of MoveNet keypoint to be displayed in processed frame
        - pred_conf: Minimum confidence of classified pose to be displayed in processed frame
        - draw_movenet_skeleton: Boolean for whether to draw MoveNet skeletons
        - blur_faces: Boolean for whether to blur a persons face
    '''
    height, width = frame.shape[:2]

    # Variables for drawing MoveNet skeleton and Classifying box
    h_box = np.sqrt(0.0036854 * width * height)
    w_box = 1.7765 * h_box
    x_0 = w_box / 2
    y_0 = h_box / 2


    movenet_time_start = time.time()

    # Reshape image for processing
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192,192)
    input_image = tf.cast(img, dtype=tf.int32)

    # MoveNet detections
    res = MOVENET(input_image)
    keypoints_with_scores = res['output_0'].numpy()[:,:,:51].reshape((6,17,3))

    for i in range(len(keypoints_with_scores)):
        if i >= max_people:
            break
        # Note: person is normalised keypoints, but keypoints_with_scores represent the actual coordinates on the frame
        person = keypoints_with_scores[i]
        temp = person.reshape(1, 51)
        temp = scaler.transform(temp)

        confidence = sum([person[j][2] for j in range(17)])/17
        if confidence > mn_conf:
            classifier_keypoints = np.array([temp[0][idx] for idx in IDX_TO_KEEP]).reshape(1, len(IDX_TO_KEEP))      # For classifier
            kp_with_scores = person.copy()                                                                           # For drawing skeleton                
            M = get_affine_transform_to_fixed_sizes_with_padding((height, width), (192, 192))
            M = np.vstack((M, [0, 0, 1]))
            M_inv = np.linalg.inv(M)[:2]
            xy_keypoints = kp_with_scores[:, :2] * 192
            xy_keypoints = cv2.transform(np.array([xy_keypoints]), M_inv)[0]
            kp_with_scores = np.hstack((xy_keypoints, kp_with_scores[:, 2:]))
            coords = kp_with_scores.flatten()

            frame = np.ascontiguousarray(frame, dtype=np.uint8) # Resolve errors in drawing

            movenet_time_end = time.time()
            movenet_time = movenet_time_end - movenet_time_start

            # Rendering
            if blur_faces:
                face_box_time, blurring_time = blur_face(frame, coords)
                if face_box_time == -1:
                    return frame, -1, -1, -1, -1
            if draw_movenet_skeleton:
                draw_time_2 = draw_skeleton(frame, kp_with_scores, kp_conf, x_0)
            model_time, draw_time_1 = classifier_prediction_for_person(classifier_keypoints, frame, pred_conf, coords, x_0, y_0, n_features=len(IDX_TO_KEEP))
    
    try:
        rendering_time = draw_time_1 + blurring_time + draw_time_2
        classifying_time = movenet_time + model_time
        face_detection_time = movenet_time + face_box_time
        return frame, rendering_time, classifying_time, face_detection_time, movenet_time
    except UnboundLocalError:
        return frame, -1, -1, -1, -1

