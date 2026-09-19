import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n-seg.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = model.track(
        source=frame,
        classes=[0],
        persist=True,
        verbose=False
    )

    for r in result:
        annotated_frame = frame.copy()

        if (r.masks is not None and
            r.boxes is not None and
                r.boxes.id is not None):

            masks = r.masks.data.numpy()
            boxes = r.boxes.xyxy.numpy()
            ids = r.boxes.id.numpy()

            for i, mask in enumerate(masks):
                person_id = ids[i]
                x1, y1, x2, y2 = boxes[i]

                mask_resize = cv2.resize(

                    mask.astype(np.uint8)*255,
                    (frame.shape[1], frame.shape[0])
                )

                contours, _ = cv2.findContours(
                    mask_resize,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                cv2.drawContours(annotated_frame, contours, -1, (0, 0, 255), 2)
                cv2.putText(annotated_frame,
                f'Shakhs Number: {person_id}',
                (int(x1), int(y1)-10),
                cv2.FONT_ITALIC,
                1,
                (0, 0, 255),
                7
                )


    max_display_width = 1200
    max_display_height = 700
    scale = min(max_display_width / annotated_frame.shape[1],
                max_display_height / annotated_frame.shape[0], 1)
    display_frame = cv2.resize(
        annotated_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    cv2.imshow("segmentation", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
