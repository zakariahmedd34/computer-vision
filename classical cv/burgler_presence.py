
import os

import cv2 as cv


cap = cv.VideoCapture(0)
save_dir = os.path.join(os.path.dirname(__file__), 'saved_images')
os.makedirs(save_dir, exist_ok=True)

frames = []
gap = 5
count = 0
# continous read
while True:
    ret, frame = cap.read()

    # nothing comes from the camera
    if not ret:
        break

    # store the frame in the dictionary

    # convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frames.append(gray)

    if len(frames) > gap + 1:
        frames.pop(0)

    cv.putText(frame, f'Frame count: {count}', (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if len(frames) > gap:
        diff = cv.absdiff(frames[0], frames[-1])
        cv.imshow('Difference', diff)

        # if the diff more than 30 and less than 255
        _, thresh = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)

        contours, _ = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv.contourArea(c) < 20000:
                continue

            x, y, w, h = cv.boundingRect(c)

            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        motion = any(cv.contourArea(c) > 20000 for c in contours)

        if motion:
            cv.putText(frame, "motion detected!", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            image_path = os.path.join(save_dir, f'motion_frame_{count}.jpg')
            cv.imwrite(image_path, frame)
            print(f"saved: {image_path}")

        cv.imshow("motion detected", frame)
        count += 1

        if cv.waitKey(1) & 0xFF == 27:  # ESC to exite
            break


cap.release()
cv.destroyAllWindows()
