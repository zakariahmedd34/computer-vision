"""Count bottles in a video and save the video with tracking boxes.

The program reads bottle.mp4, shows the detected bottles, and displays
the total number of unique bottles. The annotated video is saved as
counting_output.avi in this folder. Press q to stop the program.
"""

import os

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture('bottle.mp4')
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
output_path = os.path.join(os.path.dirname(__file__), 'counting_output.avi')
video_writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    video_fps,
    (video_width, video_height),
)
if not video_writer.isOpened():
    raise RuntimeError("Could not open the output video writer")
unique_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, classes=[39], persist=True, verbose=False)

    annotated_frame = results[0].plot()
    if results[0].boxes and results[0].boxes.id is not None:
        import numpy as np

        ids = results[0].boxes.id.numpy()

        for oid in ids:
            unique_ids.add(oid)

    cv2.putText(annotated_frame, f"Count: {len(unique_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)
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
