from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import subprocess
import detect_glasses
import argparse
import torch
import time


YOLO_V7_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                   'hair drier', 'toothbrush']

# -------------SETTINGS--------------
page_title = "Object Detection"
layout = "wide"

st.set_page_config(page_title=page_title, layout=layout,
                   page_icon='static/img/favicon.ico', )

st.markdown("""
    <style>
        #MainMenu, header, footer {visibility: hidden;}

        /* This code gets the first element on the sidebar,
        and overrides its default styling */
        section[data-testid="stSidebar"] div:first-child {
            top: 0;
            height: 10vh;
        }
        .reportview-container {
            margin-top: -2em;
        }

        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)


# ------------HEADER-----------

_, col1, _ = st.columns([3, 1, 3])
with col1:
    # st.image('static/img/logo.png',
    #          width=200)
    pass

st.write("")
st.write("")
selected = option_menu(None, ["Analyze", "About", "Contact"],
                       icons=['house', 'person', "envelope"],
                       menu_icon="cast", default_index=0, orientation="horizontal")


# -------------MAIN---------------

col3, col4, col5 = st.columns([1, 2, 2])
with col3:

    source_option = st.radio(
        'Choose data source:',
        ('Webcam', 'Uploaded Video')
    )

    classes_to_detect = st.multiselect(
        'Classes to detect:',
        YOLO_V7_CLASSES
    )

# ======================RIGHT SIDE OF PAGE===========================
with col4:
    if source_option == 'Webcam':

        st.title("Webcam Display Steamlit App")
        st.caption("Powered by OpenCV, Streamlit")

        # cap = cv2.VideoCapture(0)
        # frame_placeholder = st.empty()
        # stop_button_pressed = st.button("Stop")
        # while cap.isOpened() and not stop_button_pressed:
        #     ret, frame = cap.read()
        #     if not ret:
        #         st.write("Video Capture Ended")
        #         break
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     frame_placeholder.image(frame, channels="RGB")
        #     if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
        #         break
        # cap.release()
        # cv2.destroyAllWindows()

    else:
        file = st.file_uploader("Upload your file", type="mp3")

with col5:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    frame_placeholder2 = st.empty()
    stop_button_pressed = st.button("Stop")
        while not stop_button_pressed:
        with torch.no_grad():
            p, im0 = detect_glasses.detect(opt=opt)
        
                frame_placeholder2.image(im0, channels='RGB')
                time.sleep(0.1)





