import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict, deque
import os


model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic.mp4")
if not cap.isOpened():
    raise FileNotFoundError("Could not open traffic.mp4")

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
output_path = os.path.join(os.path.dirname(__file__), "vehicles_annotated.mp4")
video_writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    video_fps,
    (video_width, video_height),
)
if not video_writer.isOpened():
    raise RuntimeError("Could not open the output video writer")

# COCO vehicle classes: car, motorcycle, bus, and truck.
vehicle_classes = [2, 3, 5, 7]
vehicle_names = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Track the center of each vehicle.
id_map = {}
next_id = 1

trail = defaultdict(lambda: deque(maxlen=30))
appear = defaultdict(int)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = model.track(frame, classes=vehicle_classes,
                      persist=True, verbose=False)
    annotated_frame = frame.copy()

    if res[0].boxes.id is not None:
        boxes = res[0].boxes.xyxy
        ids = res[0].boxes.id.numpy()
        class_ids = res[0].boxes.cls.int().tolist()

        for box, oid, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1+x2)//2, (y1+y2)//2

            appear[oid] += 1

            if appear[oid] >= 5 and oid not in id_map:
                id_map[oid] = next_id
                next_id += 1

            if oid in id_map:
                sid = id_map[oid]
                trail[oid].append((cx, cy))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                              (255, 0, 0), 2)
                label = f"{vehicle_names[class_id]} ID: {sid}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                points = np.array(
                    trail[oid], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [
                              points], False, (0, 255, 255), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)

    video_writer.write(annotated_frame)

    max_display_width = 1200
    max_display_height = 700
    scale = min(max_display_width / annotated_frame.shape[1],
                max_display_height / annotated_frame.shape[0], 1)
    display_frame = cv2.resize(
        annotated_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    cv2.imshow("Object Tracking", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"Saved annotated video to: {output_path}")
