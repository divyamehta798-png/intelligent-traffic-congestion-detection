

import cv2
from ultralytics import YOLO
import os
import csv

model = YOLO("yolov8n.pt")

input_video = "videos/traffic_india.mp4"
output_folder = "results"
output_video = os.path.join(output_folder, "detected_traffic.mp4")
csv_file = os.path.join(output_folder, "traffic_data.csv")

os.makedirs(output_folder, exist_ok=True)

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def classify_congestion(count):
    if count <= 7:
        return "LOW"
    elif count <= 14:
        return "MEDIUM"
    else:
        return "HIGH"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("ERROR: Cannot open input video.")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_number = 0
total_detections = 0

with open(csv_file, "w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([
        "Frame", "Car", "Motorcycle",
        "Bus", "Truck", "Total_Vehicles",
        "Congestion_Level"
    ])

    print("Starting traffic analysis...")

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model(
            frame,
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        counts = {
            "Car": 0,
            "Motorcycle": 0,
            "Bus": 0,
            "Truck": 0
        }

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id not in vehicle_classes:
                    continue

                label = vehicle_classes[class_id]
                counts[label] += 1

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2),
                    (0, 255, 0), 2
                )

                cv2.putText(
                    frame, label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

        total = sum(counts.values())
        total_detections += total
        congestion = classify_congestion(total)

        if congestion == "HIGH":
            status_color = (0, 0, 255)
        elif congestion == "MEDIUM":
            status_color = (0, 165, 255)
        else:
            status_color = (0, 255, 0)

        cv2.putText(
            frame,
            f"Vehicles: {total} | Congestion: {congestion}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, status_color, 2
        )

        writer.write(frame)

        csv_writer.writerow([
            frame_number,
            counts["Car"],
            counts["Motorcycle"],
            counts["Bus"],
            counts["Truck"],
            total,
            congestion
        ])

        frame_number += 1

        if frame_number % 30 == 0:
            print(
                f"Frame: {frame_number} | "
                f"Vehicles: {total} | "
                f"Congestion: {congestion}"
            )

cap.release()
writer.release()

print("\nTraffic analysis completed!")
print("Frames processed:", frame_number)
print("Total frame-level detections:", total_detections)
print("Output video:", output_video)
print("CSV report:", csv_file)
