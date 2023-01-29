import cv2
import csv
from MoveNet_Processing_Utils import movenet_processing

# Datasets used for training, validation and test
trg_dsets = [489, 569, 581, 722, 731, 758, 807, 1219, 1260, 1301, 1373, 1378, 1392, 1790, 1843, 1954]
val_dsets = [1176, 2123]
tst_dsets = [786, 832, 925]

all_dsets = trg_dsets + val_dsets + tst_dsets

rendering_time_arr = []
classifying_time_arr = []
face_detection_time_arr = []
movenet_time_arr = []

for dataset in all_dsets:
    print("Current dataset:", dataset)

    # Get labels
    with open(f"./images/Dataset {dataset}/Labels/labels.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = [row for row in reader]


    for temp1, temp2 in labels:
        img = int(temp1)
        pose_class = int(temp2)
        if pose_class not in [1, 2, 3]: # Not standing, sitting or lying
            continue
        img_path = f"./images/Dataset {dataset}/Images/rgb_{img:04d}.png"

        '''
        update the following with relevant code needed
        '''
        frame = cv2.imread(img_path)    # BGR
        frame = frame[:,:,::-1]         # RGB

        processed, rendering_time, classifying_time, face_detection_time, movenet_time = movenet_processing(frame)

        if rendering_time != -1:
            rendering_time_arr.append(rendering_time)
            classifying_time_arr.append(classifying_time)
            face_detection_time_arr.append(face_detection_time)
            movenet_time_arr.append(movenet_time)

mean_rendering_time = sum(rendering_time_arr) / len(rendering_time_arr)
mean_classifying_time = sum(classifying_time_arr) / len(classifying_time_arr)
mean_face_detection_time = sum(face_detection_time_arr) / len(face_detection_time_arr)
mean_movenet_time = sum(movenet_time_arr) / len(movenet_time_arr)
total_time_per_frame = mean_rendering_time + mean_classifying_time + mean_face_detection_time - mean_movenet_time

print("FALL DETECTION TIME STATISTICS")
print("Rendering:", mean_rendering_time)
print("Classifying:", mean_classifying_time)
print("Face Detection:", mean_face_detection_time)
print("Movenet:", mean_movenet_time)
print("Total:", total_time_per_frame)
print(f"FPS: {round(1/total_time_per_frame, 2)}")