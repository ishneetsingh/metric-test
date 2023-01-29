import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from MoveNet_Processing_Utils import movenet_processing
import av

st.title('timing test')

def callback(frame):
    img = frame.to_ndarray(format="rgb24")

    global rendering_time_arr
    global classifying_time_arr
    global face_detection_time_arr
    global movenet_time_arr

    out_image = img.copy()

    out_image, rendering_time, classifying_time, face_detection_time, movenet_time = movenet_processing(out_image)

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

    print('-'*50)
    print("Rendering:", mean_rendering_time)
    print("Classifying:", mean_classifying_time)
    print("Face Detection:", mean_face_detection_time)
    print("Movenet:", mean_movenet_time)
    print("Total:", total_time_per_frame)
    print(f"FPS: {round(1/total_time_per_frame, 2)}")
    print('-'*50)

    return av.VideoFrame.from_ndarray(out_image, format="rgb24")

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Note: inter is for interpolating the image (to shrink it)
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    
    if width is None:
        ratio = width/float(w)
        dim = (int(w * ratio), height)
    else:
        ratio = width/float(w)
        dim = (width, int(h * ratio))

    # Resize image
    return cv2.resize(image, dim, interpolation=inter)

ctx = webrtc_streamer(
    key="real-time",
    video_frame_callback=callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    # For Deploying
    rtc_configuration={
            "iceServers": [
        {
            "urls": "stun:openrelay.metered.ca:80",
        },
        {
            "urls": "turn:openrelay.metered.ca:80",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443?transport=tcp",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        ]
    }
)

if ctx.state.playing:
    rendering_time_arr = []
    classifying_time_arr = []
    face_detection_time_arr = []
    movenet_time_arr = []