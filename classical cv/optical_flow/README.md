# Live Optical Flow Visualizer

A small educational app that shows, side by side, **image gradients** and
**sparse optical flow** from a live webcam. It uses only OpenCV and NumPy and
runs on a laptop CPU (no GPU needed).

| Window | What it shows |
|---|---|
| **Original Camera** | Live feed and the current processing FPS |
| **Image Gradients** | Sobel `Ix` (horizontal), `Iy` (vertical), and gradient magnitude |
| **Optical Flow** | Green arrows showing how tracked points moved since the last frame |

## Installation

Requires Python 3.8+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Use a different camera (e.g. an external USB webcam):

```bash
python main.py --camera 1
```

### Controls

| Key | Action |
|---|---|
| `Q` or `Esc` | Quit |
| `R` | Reset tracking (detect fresh feature points) |

### Adjustable parameters

All are optional command-line flags (`python main.py --help`):

| Flag | Default | Meaning |
|---|---|---|
| `--camera` | `0` | Camera index |
| `--fps` | `0` | Max frames processed per second (e.g. `3`); `0` = unlimited |
| `--width` | `640` | Processing width (height keeps aspect ratio). Lower = faster |
| `--sobel-ksize` | `3` | Sobel kernel size (1, 3, 5, 7) |
| `--max-corners` | `200` | Maximum number of tracked points |
| `--min-points` | `60` | Re-detect corners when fewer than this remain |
| `--win-size` | `21` | Lucas–Kanade window size (pixels) |
| `--levels` | `3` | Pyramid levels (more = handles larger motion) |
| `--fb-threshold` | `1.0` | Max forward–backward error (pixels) to keep a track |
| `--arrow-scale` | `3.0` | Arrow length multiplier for visibility |

Further defaults (corner quality, min distance, etc.) live in `TrackerParams`
in `processing.py`.

## How it works

**Gradients vs. optical flow** — two different things:

- **Gradients** describe how brightness changes *across space* within one
  frame. `Ix` is positive (red) where the image gets brighter to the right and
  negative (blue) where it gets darker; `Iy` is the same going downward. The
  magnitude `sqrt(Ix² + Iy²)` highlights edges. Colours are normalised for
  display only.
- **Optical flow** estimates how points *move between two frames* over time.

The pipeline for each frame:

1. Resize to the processing width and convert to grayscale.
2. Compute `Ix`, `Iy` with `cv2.Sobel` and their magnitude.
3. **Shi–Tomasi** (`cv2.goodFeaturesToTrack`) picks corners — points with strong
   gradients in two directions, which are the only ones that can be tracked
   reliably.
4. **Pyramidal Lucas–Kanade** (`cv2.calcOpticalFlowPyrLK`) finds where each
   corner moved. The image pyramid lets it handle motions larger than the
   search window.
5. Unreliable tracks are rejected: LK must report success, the point must stay
   inside the image, and a **forward–backward check** (track forward, then back)
   must return within `--fb-threshold` pixels of the start.
6. When fewer than `--min-points` points survive, new corners are detected away
   from existing ones.

## Project structure

```
main.py           camera capture, loop, windows, keyboard controls
processing.py     gradients, Shi–Tomasi detection, Lucas–Kanade tracking
visualization.py  gradient panels, flow arrows, labels, FPS overlay
requirements.txt
README.md
```

## Limitations

Lucas–Kanade assumes **brightness constancy** (a point keeps the same
brightness as it moves), **small motion**, and **local smoothness**. When these
break, tracking suffers:

- **Motion blur** — fast movement or long exposure in dim light smears texture,
  so corners vanish and tracks are lost.
- **Low texture** — plain walls or uniform surfaces have weak gradients; there
  are few corners to detect and motion is ambiguous (the *aperture problem*:
  along a straight edge only motion perpendicular to it can be measured).
- **Large motion** — if something moves more than the pyramid can cover in one
  frame, tracks jump to the wrong place or fail. Increase `--levels` /
  `--win-size`, or lower `--width`.
- **Brightness changes** — auto-exposure, flicker, or shadows violate
  brightness constancy and produce false or rejected tracks.
- **Sparse only** — flow is computed at selected corners, not every pixel.
- Camera shake moves the whole scene, so every arrow moves together.

## Troubleshooting

- *"could not open camera"* — check the camera is connected and not in use by
  another app (Zoom, Teams, browser), or try `--camera 1`.
- Low FPS — reduce `--width` (e.g. `--width 480`) or `--max-corners`.
