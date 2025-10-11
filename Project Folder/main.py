from ultralytics import YOLO
import cv2
import numpy as np
import streamlit as st
from sort.sort import *
import util
from util import get_car, read_license_plate, write_csv
import tempfile
import os
import pandas as pd

st.set_page_config(
    page_title="ANPR Vehicle Tracker",
    page_icon="",
    layout="wide"
)

# ---- Simple CSS Styling ----
st.markdown("""
<style>
body {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
}

h1, h2, h3 {
    color: #00ff99;
    text-align: center;
}

.stButton>button {
    background-color: #00ff99;
    color: #000;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 1rem;
}

.stProgress>div>div>div>div {
    background-color: #00ff99;
}

.stFileUploader>div {
    background-color: #1f1f1f;
    border-radius: 8px;
    padding: 10px;
    border: 2px solid #00ff99;
}

/* Video */
img {
    width: 90%;
    height: 90%;
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 8px;
}

/* Table */
.dataframe {
    width: 95%;
    margin-left: auto;
    margin-right: auto;
    border-collapse: collapse;
}

.dataframe th {
    background-color: #00ff99;
    color: #000;
    padding: 6px;
    text-align: center;
}

.dataframe td {
    background-color: #1f1f1f;
    color: #fff;
    padding: 6px;
    text-align: center;
}

.dataframe tr:hover td {
    background-color: #333333;
}
</style>
""", unsafe_allow_html=True)

st.title("Smart Traffic Management with ANPR and ATCC")
st.subheader("Upload a video to perform real-time vehicle detection & license plate recognition")

# ---- Upload video ----
uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
if uploaded_file is None:
    st.warning("⚠️ Please upload a video to start detection")
    st.stop()

# ---- Save uploaded file temporarily ----
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(uploaded_file.read())
video_path = tfile.name

results = {}
mot_tracker = Sort()

# ---- Load models ----
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/license_plate_detector.pt')
vehicles = [2, 3, 5, 7]

# ---- Video capture ----
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ---- Placeholders and progress bar ----
frame_placeholder = st.empty()
progress_bar = st.progress(0)

# ---- Vehicle log dictionary ----
vehicle_log = {}  # car_id -> {license_number, license_number_score}

# ---- Output video ----
output_path = os.path.join(os.getcwd(), "output.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ---- Frame processing ----
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}

        # ---- Detect vehicles ----
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # ---- Track vehicles ----
        track_ids = mot_tracker.update(np.asarray(detections_))

        # ---- Detect license plates ----
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(
                    license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                )
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

                    # ---- Update vehicle log if new confidence is higher ----
                    if car_id not in vehicle_log or license_plate_text_score > vehicle_log[car_id]['license_number_score']:
                        vehicle_log[car_id] = {
                            'license_number': license_plate_text,
                            'license_number_score': license_plate_text_score,
                            'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                            'license_plate_bbox': [x1, y1, x2, y2],
                            'license_plate_bbox_score': score,
                            'frame_nmr': frame_nmr
                        }

        # ---- Draw bounding boxes ----
        for car_id, info in results[frame_nmr].items():
            x1c, y1c, x2c, y2c = info['car']['bbox']
            cv2.rectangle(frame, (int(x1c), int(y1c)), (int(x2c), int(y2c)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {car_id}", (int(x1c)+5, max(int(y1c)-5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 255, 0), 2)

            x1l, y1l, x2l, y2l = info['license_plate']['bbox']
            cv2.rectangle(frame, (int(x1l), int(y1l)), (int(x2l), int(y2l)), (0, 0, 255), 2)
            text = info['license_plate']['text']
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = int((x1l + x2l - tw) / 2)
            text_y = max(int(y1l - 10), th)
            cv2.rectangle(frame, (text_x - 5, text_y - th - 5), (text_x + tw + 5, text_y + 5), (255, 255, 255), -1)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # ---- Show live frame 90% size ----
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # ---- Write frame to output ----
        out.write(frame)

        # ---- Update progress ----
        progress_bar.progress(min(frame_nmr / frame_count, 1.0))

# ---- Release resources ----
cap.release()
out.release()

# ---- Write CSV results ----
csv_output = './test.csv'
df = pd.DataFrame([
    {
        'frame_nmr': v['frame_nmr'],
        'car_id': int(k),
        'car_bbox': v['car_bbox'],
        'license_plate_bbox': v['license_plate_bbox'],
        'license_plate_bbox_score': v['license_plate_bbox_score'],
        'license_number': v['license_number'],
        'license_number_score': v['license_number_score']
    } for k, v in vehicle_log.items()
])
df.to_csv(csv_output, index=False)
util.interpolate_and_write_csv(csv_output, './test_interpolated.csv')

st.success("✅ Vehicle detection and ANPR processing completed!")

# ---- Display video slightly smaller ----
st.video(output_path)

# ---- Display vehicle log table ----
st.subheader("Vehicle License Log")
st.dataframe(df)
st.markdown(f"Output video saved at: `{output_path}`")
