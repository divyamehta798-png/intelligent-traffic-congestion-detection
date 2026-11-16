

import streamlit as st
import pandas as pd
import os

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Intelligent Traffic Congestion Detection",
    page_icon="🚦",
    layout="wide"
)

# --------------------------------------------------
# 2. FILE PATHS
# --------------------------------------------------

CSV_PATH = "results/traffic_data.csv"
VIDEO_PATH = "results/detected_traffic.mp4"

# --------------------------------------------------
# 3. DASHBOARD HEADER
# --------------------------------------------------

st.title("🚦 Intelligent Traffic Congestion Detection System")

st.markdown(
    "### AI-powered traffic monitoring using YOLOv8 and OpenCV"
)

st.markdown(
    "This software prototype detects vehicles from recorded "
    "traffic video, estimates congestion levels, and displays "
    "traffic analysis through an interactive dashboard."
)

st.divider()

# --------------------------------------------------
# 4. CHECK INPUT DATA
# --------------------------------------------------

if not os.path.exists(CSV_PATH):
    st.error(
        "Traffic data not found. Please run detection.py first."
    )
    st.stop()

try:
    df = pd.read_csv(CSV_PATH)
except Exception as error:
    st.error(f"Unable to read traffic data: {error}")
    st.stop()

required_columns = [
    "Frame",
    "Car",
    "Motorcycle",
    "Bus",
    "Truck",
    "Total_Vehicles",
    "Congestion_Level"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "The CSV file is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if df.empty:
    st.warning("The traffic data file is empty.")
    st.stop()

# --------------------------------------------------
# 5. DATA PROCESSING
# --------------------------------------------------

total_frames = len(df)

average_vehicles = df["Total_Vehicles"].mean()

maximum_vehicles = df["Total_Vehicles"].max()

congestion_counts = (
    df["Congestion_Level"]
    .value_counts()
    .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
)

high_congestion_frames = int(
    congestion_counts.get("HIGH", 0)
)

vehicle_columns = [
    "Car",
    "Motorcycle",
    "Bus",
    "Truck"
]

vehicle_totals = df[vehicle_columns].sum()

# --------------------------------------------------
# 6. TRAFFIC OVERVIEW
# --------------------------------------------------

st.subheader("📊 Traffic Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Frames Analysed",
    total_frames
)

col2.metric(
    "Average Vehicles / Frame",
    f"{average_vehicles:.2f}"
)

col3.metric(
    "Maximum Vehicles / Frame",
    int(maximum_vehicles)
)

col4.metric(
    "High Congestion Frames",
    high_congestion_frames
)

st.divider()

# --------------------------------------------------
# 7. CONGESTION LEVEL DISTRIBUTION
# --------------------------------------------------

st.subheader("🚦 Congestion Level Distribution")

distribution = pd.DataFrame({
    "Frames": congestion_counts
})

st.bar_chart(distribution)

st.caption(
    "Congestion categories are based on configurable "
    "vehicle-count thresholds used in this prototype."
)

# --------------------------------------------------
# 8. VEHICLE CATEGORY ANALYSIS
# --------------------------------------------------

st.subheader("🚗 Vehicle Category Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Vehicle detections by category**")
    st.bar_chart(vehicle_totals)

with col2:
    st.markdown("**Congestion summary**")

    summary = pd.DataFrame({
        "Congestion Level": ["LOW", "MEDIUM", "HIGH"],
        "Frames": [
            int(congestion_counts["LOW"]),
            int(congestion_counts["MEDIUM"]),
            int(congestion_counts["HIGH"])
        ]
    })

    st.dataframe(
        summary,
        hide_index=True,
        width="stretch"
    )

st.divider()

# --------------------------------------------------
# 9. VEHICLE COUNT OVER TIME
# --------------------------------------------------

st.subheader("📈 Vehicle Count Over Time")

chart_data = df.set_index("Frame")[["Total_Vehicles"]]

st.line_chart(chart_data)

# --------------------------------------------------
# 10. AI VEHICLE DETECTION VIDEO
# --------------------------------------------------

st.subheader("🎥 AI Vehicle Detection Output")

if os.path.exists(VIDEO_PATH):
    st.video(VIDEO_PATH)
else:
    st.warning(
        "Annotated video not found. "
        "Please run detection.py first."
    )

# --------------------------------------------------
# 11. FRAME-BY-FRAME TRAFFIC RECORDS
# --------------------------------------------------

st.divider()

st.subheader("📋 Frame-by-Frame Traffic Records")

st.dataframe(
    df.head(20),
    width="stretch"
)

# --------------------------------------------------
# 12. DOWNLOAD TRAFFIC REPORT
# --------------------------------------------------

st.subheader("📥 Download Traffic Analysis")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Traffic Analysis CSV",
    data=csv_data,
    file_name="traffic_analysis_report.csv",
    mime="text/csv"
)

# --------------------------------------------------
# 13. PROJECT INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("ℹ️ Project Information")

st.write(
    "**Project:** Intelligent Traffic Congestion "
    "Detection and Monitoring System Using Computer Vision"
)

st.write("**Vehicle Detection Model:** YOLOv8n")

st.write("**Computer Vision Library:** OpenCV")

st.write("**Dashboard Framework:** Streamlit")

st.write("**Data Processing:** Python and Pandas")

st.caption(
    "Prototype limitation: Congestion levels are estimated "
    "using configurable vehicle-count thresholds. They are "
    "not calibrated against official road-capacity standards. "
    "Results are based on the recorded input video and do not "
    "represent live traffic monitoring."
)
