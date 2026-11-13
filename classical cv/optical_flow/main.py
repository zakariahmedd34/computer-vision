"""Live optical-flow visualizer.

Windows:
  1. Original Camera  - live feed + processing FPS
  2. Image Gradients  - Sobel Ix, Iy and gradient magnitude
  3. Optical Flow     - Lucas-Kanade motion arrows on tracked corners

Keys: Q / Esc = quit, R = reset tracking.
"""

import argparse
import sys
import time

import cv2

from processing import FlowTracker, TrackerParams, compute_gradients
from visualization import build_gradient_panel, draw_flow, draw_fps

WIN_CAMERA = "Original Camera"
WIN_GRADIENTS = "Image Gradients"
WIN_FLOW = "Optical Flow"

MAX_FAILED_READS = 30  # give up after this many consecutive failed frame reads


def parse_args():
    ap = argparse.ArgumentParser(description="Educational optical flow visualizer.")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    ap.add_argument("--fps", type=float, default=0,
                    help="max frames processed per second, e.g. 3 (default: 0 = as fast as possible)")
    ap.add_argument("--width", type=int, default=640,
                    help="processing width in pixels; height keeps aspect ratio (default: 640)")
    ap.add_argument("--sobel-ksize", type=int, default=3, choices=[1, 3, 5, 7],
                    help="Sobel kernel size (default: 3)")
    ap.add_argument("--max-corners", type=int, default=200, help="max tracked points")
    ap.add_argument("--min-points", type=int, default=60,
                    help="re-detect features when fewer points remain")
    ap.add_argument("--win-size", type=int, default=21, help="Lucas-Kanade window size")
    ap.add_argument("--levels", type=int, default=3, help="Lucas-Kanade pyramid levels")
    ap.add_argument("--fb-threshold", type=float, default=1.0,
                    help="max forward-backward error in pixels to accept a track")
    ap.add_argument("--arrow-scale", type=float, default=3.0,
                    help="multiply arrow length for visibility")
    return ap.parse_args()


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    new_h = int(round(h * width / w))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def window_closed(name):
    """True if the user closed the window with its [X] button."""
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: could not open camera {args.camera}. "
              f"Is it connected and not used by another app? Try --camera 1.",
              file=sys.stderr)
        return 1

    tracker = FlowTracker(TrackerParams(
        max_corners=args.max_corners,
        min_points=args.min_points,
        win_size=args.win_size,
        max_level=args.levels,
        fb_threshold=args.fb_threshold,
    ))

    for name in (WIN_CAMERA, WIN_GRADIENTS, WIN_FLOW):
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    print("Running. Press Q or Esc to quit, R to reset tracking.")
    failed_reads = 0
    fps = 0.0
    last_time = time.perf_counter()
    # Frame-rate cap: process one frame every `interval` seconds.
    interval = 1.0 / args.fps if args.fps > 0 else 0.0
    next_process = 0.0

    try:
        while True:
            # grab() pulls a frame from the camera without decoding it. We grab
            # every frame so the camera buffer never fills with stale images.
            ok = cap.grab()
            if not ok:
                failed_reads += 1
                if failed_reads >= MAX_FAILED_READS:
                    print("Error: camera stopped delivering frames.", file=sys.stderr)
                    break
                cv2.waitKey(10)
                continue
            failed_reads = 0

            # Not time to process yet: keep windows responsive and skip this frame.
            if interval and time.perf_counter() < next_process:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    tracker.reset()
                    print("Tracking reset.")
                continue
            next_process = time.perf_counter() + interval

            ok, frame = cap.retrieve()  # decode the latest grabbed frame
            if not ok or frame is None:
                continue

            # Work at a fixed, smaller resolution: faster and consistent results.
            frame = resize_to_width(frame, args.width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Processing ---
            ix, iy, mag = compute_gradients(gray, ksize=args.sobel_ksize)
            old_pts, new_pts = tracker.update(gray)

            # FPS (exponential moving average for a stable readout).
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

            # --- Visualization ---
            cam_vis = frame.copy()
            draw_fps(cam_vis, fps)
            cv2.imshow(WIN_CAMERA, cam_vis)
            cv2.imshow(WIN_GRADIENTS, build_gradient_panel(ix, iy, mag))
            cv2.imshow(WIN_FLOW, draw_flow(frame, old_pts, new_pts,
                                           arrow_scale=args.arrow_scale))

            # --- Keyboard ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = Esc
                break
            if key in (ord("r"), ord("R")):
                tracker.reset()
                print("Tracking reset.")
            if any(window_closed(n) for n in (WIN_CAMERA, WIN_GRADIENTS, WIN_FLOW)):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
