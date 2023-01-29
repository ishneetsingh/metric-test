import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
from insightface.app import FaceAnalysis
import time

@st.cache(allow_output_mutation=True)
def load_insightface():
    app = FaceAnalysis(allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

INSIGHTFACE = load_insightface()
st.title('timing test')

def callback(frame):
    img = frame.to_ndarray(format="rgb24")

    global face_detection_time_arr

    out_img = img.copy()

    start_time = time.time()
    faces = INSIGHTFACE.get(out_img)
    end_time = time.time()

    face_detection_time_arr.append(end_time - start_time)
    
    for face in faces:
        # Blurring
        x1  = int(face['bbox'][0])
        y1  = int(face['bbox'][1])
        x2 = int(face['bbox'][2])
        y2 = int(face['bbox'][3])

        roi = out_img[y1:y2, x1:x2]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        out_img[y1:y2, x1:x2] = roi
    
    mean_face_detecion_time = sum(face_detection_time_arr) / len(face_detection_time_arr)
    print('-'*50)
    print("Face Detection Time:", mean_face_detecion_time)
    print('-'*50)

    return av.VideoFrame.from_ndarray(out_img, format="rgb24")

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
    face_detection_time_arr = []