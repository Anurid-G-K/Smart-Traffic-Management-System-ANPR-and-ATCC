Smart Traffic Management with ANPR and ATCC

1. Project Overview

This project implements an end-to-end traffic analysis system combining Automatic Number Plate Recognition (ANPR) and Automatic Traffic Counting and Classification (ATCC).
It uses YOLOv8 for vehicle and license plate detection, SORT for tracking, and an OCR-based recognition module for reading number plates.
The project also includes a Streamlit-based dashboard for visualization and control.


2. Key Features

Vehicle Detection and Tracking
Detects and tracks cars, buses, trucks, and two-wheelers using YOLOv8 and SORT.

License Plate Recognition (ANPR)
Detects and reads license plate numbers through OCR-based text extraction.

Frame Interpolation
Smooths tracking data and fills missing frames for better visualization.

Automated CSV Logging
Exports detection data to test.csv and interpolated data to test_interpolated.csv.

Streamlit Dashboard
Provides a graphical user interface for uploading videos, monitoring detections, and viewing results.

Visualization Module
Generates annotated videos and logs results automatically.

Clean UI Design
Custom CSS for a dark, professional, and responsive dashboard layout.



3. System Architecture

Input Video
     ↓
 YOLOv8 Object Detection
     ↓
 Vehicle Tracking (SORT)
     ↓
 License Plate Detection (YOLOv8)
     ↓
 OCR-based Plate Recognition
     ↓
 Frame Interpolation + CSV Logging
     ↓
 Streamlit Dashboard + Visualization



4. Project Structure

automatic-number-plate-recognition-python-yolov8/
│
├── main.py                        # Streamlit-based detection and processing script
├── visualize.py                   # Visualization and video rendering module
├── util.py                        # Utility functions for detection and processing
├── sort/
│   ├── sort.py                    # SORT tracking algorithm
│   └── __init__.py
│
├── models/
│   ├── license_plate_detector.pt  # Custom YOLOv8 model for license plate detection
│   └── yolov8n.pt                 # Coco model

├── sample.mp4                     # Example input video
├── test.csv                       # Raw detection results
├── test_interpolated.csv          # Interpolated tracking data
├── output.avi                     # Final annotated video output
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentatio


5. Installation

git clone https://github.com/Anurid-G-K/Smart-Traffic-Management-System-ANPR-and-ATCC
cd Smart-Traffic-Management-System-ANPR-and-ATCC


6. Running the Application

   6.1 Launch the Streamlit Dashboard
   streamlit run main.py

   6.2 Usage Steps
   Upload a video file (MP4, AVI, or MOV format).
   The system automatically performs: Vehicle detection and tracking, License plate detection and recognition, CSV generation and interpolation, Automatic visualization of results


7. Output Files

| File                    | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| `test.csv`              | Raw frame-wise vehicle and plate detection results               |
| `test_interpolated.csv` | Interpolated tracking data for smoother continuity               |
| `output.avi`            | Annotated output video with bounding boxes and recognized plates |



8. Models Used

| Component               | Model                                      | Purpose                                       |
| ----------------------- | ------------------------------------------ | --------------------------------------------- |
| Vehicle Detection       | `yolov8n.pt`                               | Detects vehicles (cars, buses, trucks, bikes) |
| License Plate Detection | `license_plate_detector.pt`                | Detects license plates on vehicles            |
| Tracker                 | SORT (Kalman Filter + Hungarian Algorithm) | Tracks vehicles across frames                 |
| OCR                     | Tesseract-based recognition                | Reads characters from license plate crops     |

