# Intelligent Traffic Congestion Detection and Monitoring System Using Computer Vision

## 1. Project Overview

This project is a software-based traffic congestion detection and monitoring prototype developed using Python and computer vision.

It uses the YOLOv8n object detection model to identify vehicles in recorded traffic videos. The system counts detected vehicles in each frame, classifies congestion levels, generates an annotated video, and displays traffic analysis through a Streamlit dashboard.

## 2. Technologies Used

- Python 3.13.3
- YOLOv8n
- OpenCV
- Pandas
- Streamlit
- Matplotlib

## 3. Project Structure

```text
Traffic_Congestion_Detection/
│
├── videos/
│   └── traffic_india.mp4
│
├── results/
│   ├── detected_traffic.mp4
│   └── traffic_data.csv
│
├── data/
├── docs/
├── detection.py
├── app.py
├── requirements.txt
└── README.md
```

## 4. Installation

Install Python and open Command Prompt in the project directory.

Create and activate a virtual environment:

```cmd
python -m venv venv
venv\Scripts\activate
```

Install the required libraries:

```cmd
pip install -r requirements.txt
```

The YOLOv8n model weights are downloaded automatically when required if they are not already available locally.

## 5. Execution

### Step 1: Vehicle Detection and Traffic Analysis

Run:

```cmd
python detection.py
```

The program reads the input traffic video, detects vehicles, calculates frame-wise counts, classifies congestion levels, and generates output files.

### Step 2: Launch the Dashboard

Run:

```cmd
streamlit run app.py
```

Open the following address in your browser:

```text
http://localhost:8501
```

## 6. Congestion Classification

The prototype uses these configured thresholds:

| Vehicle detections per frame | Congestion |
|---|---|
| 0–7 | LOW |
| 8–14 | MEDIUM |
| 15 or more | HIGH |

These are experimental prototype thresholds, not official traffic-engineering standards.

## 7. Output Files

**detected_traffic.mp4:** Annotated traffic video displaying detected vehicles with bounding boxes.

**traffic_data.csv:** Frame-wise vehicle counts and congestion classifications.

## 8. Observed Results

The test video was processed successfully.

- Frames processed: 484
- Total frame-level detections: 5,982
- Average detections per frame: 12.36
- Maximum detections in a frame: 19
- LOW congestion frames: 17
- MEDIUM congestion frames: 385
- HIGH congestion frames: 82

Vehicle category detections:

- Cars: 3,872
- Motorcycles: 414
- Buses: 1,248
- Trucks: 448

Note: These are frame-level detections, not unique vehicle counts. The same vehicle can be detected in multiple frames.

## 9. Limitations

- The current implementation uses recorded video, not live camera input.
- Individual vehicles are not tracked across frames.
- Detection accuracy has not been evaluated against a labelled ground-truth dataset.
- The system does not control traffic signals.

## 10. Future Scope

- Real-time traffic monitoring.
- Vehicle tracking and unique vehicle counting.
- Speed and queue-length estimation.
- Evaluation using labelled traffic datasets.
- Integration with intelligent traffic management systems.

## 11. Project Information

**Project Title:** Intelligent Traffic Congestion Detection and Monitoring System Using Computer Vision

**Student:** Divya Mehta

**Roll Number:** 2300320210022

**Department:** Electrical and Electronics Engineering

**Institution:** ABES Engineering College, Ghaziabad

**Mentor:** Dr. Pavan Khetrapal