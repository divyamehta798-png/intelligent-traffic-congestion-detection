import subprocess
import imageio_ffmpeg
import streamlit as st
import pandas as pd
import os
import tempfile
import cv2
from ultralytics import YOLO

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Intelligent Traffic Congestion Detection",
    page_icon="🚦",
    layout="wide"
)

CSV_PATH = "results/traffic_data.csv"
VIDEO_PATH = "results/detected_traffic.mp4"

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Prototype thresholds: LOW 0-7, MEDIUM 8-14, HIGH 15+
def congestion_level(total):
    if total <= 7:
        return "LOW"
    elif total <= 14:
        return "MEDIUM"
    return "HIGH"


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


def process_video(video_path, confidence, max_frames):
    """Process uploaded video and return CSV records plus annotated MP4 bytes."""
    model = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("The video has invalid dimensions.")

    temp_output = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    )
    output_path = temp_output.name
    temp_output.close()

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    records = []
    frame_number = 0
    progress = st.progress(0)
    status = st.empty()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Match detection.py: first processed frame is numbered 0.
            current_frame = frame_number
            frame_number += 1

            # 0 means process the complete video; otherwise stop at the chosen limit.
            if max_frames > 0 and frame_number > max_frames:
                break

            prediction = model.predict(
                frame,
                conf=confidence,
                imgsz=640,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False
            )[0]

            counts = {
                "Car": 0,
                "Motorcycle": 0,
                "Bus": 0,
                "Truck": 0
            }

            if prediction.boxes is not None:
                for class_id in prediction.boxes.cls.tolist():
                    class_id = int(class_id)
                    vehicle_name = VEHICLE_CLASSES.get(class_id)
                    if vehicle_name:
                        counts[vehicle_name] += 1

            total = sum(counts.values())
            level = congestion_level(total)

            records.append({
                "Frame": current_frame,
                "Car": counts["Car"],
                "Motorcycle": counts["Motorcycle"],
                "Bus": counts["Bus"],
                "Truck": counts["Truck"],
                "Total_Vehicles": total,
                "Congestion_Level": level
            })

            annotated_frame = prediction.plot()

            # Add the same total-count and congestion overlay as detection.py.
            if level == "HIGH":
                status_color = (0, 0, 255)
            elif level == "MEDIUM":
                status_color = (0, 165, 255)
            else:
                status_color = (0, 255, 0)

            cv2.putText(
                annotated_frame,
                f"Vehicles: {total} | Congestion: {level}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )
            writer.write(annotated_frame)

            if total_frames > 0:
                progress_value = min(frame_number / total_frames, 1.0)
            elif max_frames > 0:
                progress_value = min(frame_number / max_frames, 1.0)
            else:
                progress_value = 0

            progress.progress(progress_value)
            status.text(f"Processing frame {frame_number}...")

    finally:
        cap.release()
        writer.release()

    progress.progress(1.0)
    status.success(f"Processing complete: {len(records)} frames analyzed.")

    if not records:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise ValueError("No frames were processed.")

    df = pd.DataFrame(records)

# Convert the generated video to browser-compatible H.264 MP4
    h264_temp = tempfile.NamedTemporaryFile(
        suffix="_h264.mp4",
        delete=False
    )
    h264_path = h264_temp.name
    h264_temp.close()

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i", output_path,
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                h264_path
            ],
            check=True,
            capture_output=True,
            text=True
        )

        with open(h264_path, "rb") as output_file:
            video_bytes = output_file.read()

    finally:
        for path in [output_path, h264_path]:
            try:
                os.remove(path)
            except OSError:
                pass

    return df, video_bytes

def show_analysis(df, video_bytes=None):
    required_columns = [
        "Frame", "Car", "Motorcycle", "Bus", "Truck",
        "Total_Vehicles", "Congestion_Level"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error("Missing columns in traffic data: " + ", ".join(missing))
        return

    if df.empty:
        st.warning("No traffic records are available.")
        return

    total_frames = len(df)
    average_vehicles = df["Total_Vehicles"].mean()
    maximum_vehicles = df["Total_Vehicles"].max()

    congestion_counts = (
        df["Congestion_Level"]
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
    )

    high_frames = int(congestion_counts.get("HIGH", 0))
    vehicle_columns = ["Car", "Motorcycle", "Bus", "Truck"]
    vehicle_totals = df[vehicle_columns].sum()

    st.subheader("📊 Traffic Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames Analysed", total_frames)
    col2.metric("Average Vehicles / Frame", f"{average_vehicles:.2f}")
    col3.metric("Maximum Vehicles / Frame", int(maximum_vehicles))
    col4.metric("High Congestion Frames", high_frames)

    st.divider()
    st.subheader("🚦 Congestion Level Distribution")
    st.bar_chart(pd.DataFrame({"Frames": congestion_counts}))
    st.caption(
        "Prototype categories: LOW (0–7 vehicles), MEDIUM (8–14), "
        "HIGH (15 or more). These thresholds are configurable and "
        "not calibrated to official road-capacity standards."
    )

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
        st.dataframe(summary, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("📈 Vehicle Count Over Time")
    st.line_chart(df.set_index("Frame")[["Total_Vehicles"]])

    st.subheader("📋 Frame-by-Frame Traffic Records")
    st.dataframe(df.head(50), hide_index=True, use_container_width=True)

    st.subheader("📥 Download Results")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download Traffic Analysis CSV",
            data=csv_bytes,
            file_name="traffic_analysis_report.csv",
            mime="text/csv"
        )
    if video_bytes:
        with col2:
            st.download_button(
                "Download Annotated Video",
                data=video_bytes,
                file_name="detected_traffic.mp4",
                mime="video/mp4"
            )
        st.subheader("🎥 AI Vehicle Detection Output")
        st.video(video_bytes, format="video/mp4")
# --------------------------------------------------
# 2. DASHBOARD HEADER
# --------------------------------------------------
st.title("🚦 Intelligent Traffic Congestion Detection System")
st.markdown("### AI-powered traffic monitoring using YOLOv8 and OpenCV")
st.markdown(
    "Upload a recorded traffic video to manually start vehicle detection, "
    "or view previously generated traffic analysis."
)

tab_manual, tab_existing = st.tabs(["📤 Manual Video Analysis", "📊 Existing Results"])

# --------------------------------------------------
# 3. MANUAL VIDEO UPLOAD AND PROCESSING
# --------------------------------------------------
with tab_manual:
    st.subheader("Manual Operation: Upload and Start Detection")

    uploaded_video = st.file_uploader(
        "Upload traffic video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Choose a recorded traffic video from your device."
    )

    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

    max_frames = st.number_input(
        "Maximum frames to process (0 = entire video)",
        min_value=0,
        max_value=10000,
        value=0,
        step=100,
        help="Processing an entire long video may take several minutes."
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("▶️ Start Detection", type="primary"):
            input_temp = tempfile.NamedTemporaryFile(
                suffix=os.path.splitext(uploaded_video.name)[1],
                delete=False
            )
            input_path = input_temp.name
            input_temp.write(uploaded_video.getbuffer())
            input_temp.close()

            try:
                with st.spinner("Loading YOLOv8 and processing video..."):
                    result_df, result_video = process_video(
                        input_path,
                        confidence,
                        int(max_frames)
                    )

                st.session_state["manual_df"] = result_df
                st.session_state["manual_video"] = result_video
                st.session_state["manual_filename"] = uploaded_video.name
                st.success("Video analysis completed successfully!")

            except Exception as error:
                st.error(f"Processing failed: {error}")

            finally:
                try:
                    os.remove(input_path)
                except OSError:
                    pass

    if "manual_df" in st.session_state:
        st.divider()
        st.subheader("Manual Detection Results")
        st.caption(
            "Source video: " + st.session_state.get("manual_filename", "Uploaded video")
        )
        show_analysis(
            st.session_state["manual_df"],
            st.session_state.get("manual_video")
        )

# --------------------------------------------------
# 4. EXISTING RESULTS
# --------------------------------------------------
with tab_existing:
    st.subheader("Previously Generated Traffic Results")

    if os.path.exists(CSV_PATH):
        try:
            existing_df = pd.read_csv(CSV_PATH)
            if existing_df.empty:
                st.warning("The existing traffic CSV is empty.")
            else:
                st.caption("Showing analysis from results/traffic_data.csv")
                existing_video_bytes = None

                if os.path.exists(VIDEO_PATH):
                    with open(VIDEO_PATH, "rb") as video_file:
                        existing_video_bytes = video_file.read()

                show_analysis(existing_df, existing_video_bytes)

        except Exception as error:
            st.error(f"Unable to load existing results: {error}")
    else:
        st.info(
            "No pre-generated traffic CSV was found. "
            "Use the Manual Video Analysis tab to upload a video and start detection."
        )

# --------------------------------------------------
# 5. PROJECT INFORMATION
# --------------------------------------------------
st.divider()
st.subheader("ℹ️ Project Information")
st.write(
    "**Project:** Intelligent Traffic Congestion Detection and Monitoring "
    "System Using Computer Vision"
)
st.write("**Vehicle Detection Model:** YOLOv8n")
st.write("**Computer Vision Library:** OpenCV")
st.write("**Dashboard Framework:** Streamlit")
st.write("**Data Processing:** Python and Pandas")
st.caption(
    "Software prototype. Vehicle detections are frame-level counts, not unique "
    "vehicle tracking. Congestion thresholds are configurable and are not "
    "validated against official road-capacity standards. This application "
    "processes uploaded recorded videos; it does not monitor live cameras."
)

