import ast
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="ANPR Visualization", page_icon="🎥", layout="wide")

# ✅ Read interpolated data instead of original CSV
results = pd.read_csv('./test_interpolated.csv')

# load video
video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)

# ✅ Ensure output directory exists (current folder)
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# ✅ Define full output video path (AVI)
output_path = os.path.join(output_dir, "out.avi")

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # AVI codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ✅ Prepare license plate numbers
license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0
frame_placeholder = st.empty()
progress_bar = st.progress(0)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        # ✅ Add translucent box with current time and date
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 110), (0, 0, 0), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        cv2.putText(frame, f"Time: {time_str}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Date: {date_str}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            car_id = int(df_.iloc[row_indx]['car_id'])

            # draw car bounding box
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 2)

            # ✅ Display vehicle ID slightly above bounding box, font +10%
            id_text = f"ID: {car_id}"
            font_scale = 0.66
            thickness = 2
            id_color = (0, 255, 0)
            id_x = int(car_x1) + 5
            id_y = max(int(car_y1) - 5, 15)
            cv2.putText(frame, id_text, (id_x, id_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, id_color, thickness)

            # draw license plate bounding box
            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # ✅ Draw OCR text with white background
            text = license_plate[car_id]['license_plate_number']
            (text_width, text_height), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                2
            )
            text_x = int((x1 + x2 - text_width) / 2)
            text_y = max(int(y1 - 10), text_height)

            cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5),
                          (text_x + text_width + 5, text_y + 5), (255, 255, 255), -1)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # write frame to output video
        out.write(frame)
        frame_count += 1

        # Resize for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (1280, 720))
        frame_placeholder.image(frame_rgb, channels="RGB")

        # update progress bar
        progress_bar.progress(min(frame_nmr / total_frames, 1.0))

out.release()
st.success(f"[INFO] Total frames written: {frame_count}")
st.video(output_path)

cap.release()
