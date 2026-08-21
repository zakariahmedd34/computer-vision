import cv2
from ultralytics import YOLO


# Use 0 for default camera

cap = cv2.VideoCapture("street2.mp4")
# cv2.namedWindow("Annotated video", cv2.WINDOW_NORMAL)

model = YOLO("yolov8n.pt")


while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame,classes=[0])
    annotated_frame = results[0].plot()

    max_display_width = 1200
    max_display_height = 400
    scale = min(max_display_width / annotated_frame.shape[1],
                max_display_height / annotated_frame.shape[0], 1)
    display_frame = cv2.resize(
        annotated_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    cv2.imshow("Annotated video", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
