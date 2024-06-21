import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# Add YOLOv7 directory to the Python path
import sys
sys.path.append("yolov7")

# Import YOLOv7 model definition
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.datasets import letterbox

# Load YOLOv7 model
@st.cache_resource
def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    return model

def get_class_names(model):
    return model.names

def generate_colors(num_classes):
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype='uint8')
    return colors

def detect_objects(image, model, device, selected_classes=None, conf_thres=0.25, iou_thres=0.45):
    # Preprocess the image
    img = letterbox(image, 640, stride=32, auto=True)[0]  # Resize to 640x640, padded
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()  # Convert to tensor
    img /= 255.0  # Normalize
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=selected_classes, agnostic=False)

    # Process detections
    colors = generate_colors(len(model.names))
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f"{model.names[int(cls)]} {conf:.2f}"
                color = colors[int(cls)].tolist()
                plot_one_box(xyxy, image, label=label, color=color, line_thickness=2)
    return image

models = {
    "Custom Glasses Detection": "glasses.pt",
    "Pretrained YOLO v7": "yolov7.pt",
}

# Streamlit application
st.title("Object Detection using Custom YOLOv7")

st.sidebar.title("Options")
option = st.sidebar.selectbox("Choose input type:", ["Web Camera", "Upload Image", "Upload Video"])
chosen_model = st.sidebar.selectbox("Choose model:", ["Custom Glasses Detection", "Pretrained YOLO v7"])

# Load model
model = load_model(models[chosen_model])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get class names
class_names = get_class_names(model)

# Add multiselect for classes
selected_classes = st.sidebar.multiselect("Select classes to detect:", class_names)

# Convert class names to class indices
selected_class_indices = [class_names.index(cls) for cls in selected_classes]

confidence = st.sidebar.select_slider("Confidence threshold:", [x/100 for x in range(0, 100, 10)])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        st.write("Detecting objects...")

        # Convert image to RGB if needed
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif image_np.shape[2] == 1:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        detected_image = detect_objects(image_np, model, device, selected_class_indices, conf_thres=confidence)
        st.image(detected_image, caption="Detected Image.", use_column_width=True)

elif option == "Web Camera":
    st.write("Web Camera")
    run_live = st.button("Run Live Detection")

    if run_live:
        stop_button_pressed = st.button("Stop")

        cap = cv2.VideoCapture(0)
        frame_window = st.image([])

        while run_live and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_objects(frame_rgb, model, device, selected_class_indices, conf_thres=confidence)
            frame_window.image(detected_frame, channels="RGB")

        cap.release()

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is not None:
        st.write("Detecting objects in video...")

        # Save uploaded video to a file
        video_path = Path("uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        cap = cv2.VideoCapture(str(video_path))
        frame_window = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_frame = detect_objects(frame_rgb, model, device, selected_class_indices, conf_thres=confidence)
            frame_window.image(detected_frame, channels="RGB")

        cap.release()
