"""Drawing helpers: gradient panels, flow arrows, labels, and FPS overlay.

All normalization here is for DISPLAY ONLY - the raw float gradients used for
processing are never modified.
"""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_label(img, text, org=(8, 22), scale=0.55, color=(255, 255, 255)):
    """Draw readable text: a dark outline under a coloured fill."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, 1, cv2.LINE_AA)


def draw_fps(img, fps):
    put_label(img, f"FPS: {fps:5.1f}", org=(8, 22), color=(0, 255, 255))


# --------------------------------------------------------------------------
# Gradient visualization
# --------------------------------------------------------------------------
def signed_to_color(d):
    """Map a signed derivative to a diverging colour image.

    black = 0 (no brightness change), red = positive, blue = negative.
    Values are scaled by a robust maximum (99th percentile of |d|) so a few
    extreme pixels do not wash out the rest of the image.
    """
    scale = np.percentile(np.abs(d), 99) + 1e-6
    v = np.clip(d / scale, -1.0, 1.0)
    out = np.zeros(d.shape + (3,), np.uint8)          # BGR
    out[..., 2] = (np.clip(v, 0, 1) * 255).astype(np.uint8)   # red   = positive
    out[..., 0] = (np.clip(-v, 0, 1) * 255).astype(np.uint8)  # blue  = negative
    return out


def magnitude_to_color(mag):
    """Map non-negative gradient magnitude to a perceptual colormap."""
    scale = np.percentile(mag, 99) + 1e-6
    norm = (np.clip(mag / scale, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def build_gradient_panel(ix, iy, mag, panel_scale=0.5):
    """Return one image with Ix | Iy | magnitude side by side, each labelled."""
    panels = [
        (signed_to_color(ix), "Ix  (d/dx)  red:+  blue:-"),
        (signed_to_color(iy), "Iy  (d/dy)  red:+  blue:-"),
        (magnitude_to_color(mag), "|grad| = sqrt(Ix^2 + Iy^2)"),
    ]
    out = []
    for img, label in panels:
        if panel_scale != 1.0:
            img = cv2.resize(img, None, fx=panel_scale, fy=panel_scale,
                             interpolation=cv2.INTER_AREA)
        put_label(img, label, scale=0.45)
        out.append(img)
    strip = np.hstack(out)

    # Footer explaining what the window shows.
    footer = np.zeros((26, strip.shape[1], 3), np.uint8)
    put_label(footer, "Gradients: brightness change ACROSS SPACE in a single frame",
              org=(8, 18), scale=0.5, color=(200, 200, 200))
    return np.vstack([strip, footer])


# --------------------------------------------------------------------------
# Optical flow visualization
# --------------------------------------------------------------------------
def draw_flow(frame, old_pts, new_pts, arrow_scale=3.0, min_motion=0.5):
    """Draw motion arrows on a copy of `frame`.

    Each arrow starts at the point's current position and points along its
    motion since the previous frame. `arrow_scale` exaggerates the length so
    small per-frame motion is visible. Points that barely moved get a dot.
    """
    vis = frame.copy()
    for (x0, y0), (x1, y1) in zip(old_pts, new_pts):
        dx, dy = x1 - x0, y1 - y0
        start = (int(round(x1)), int(round(y1)))
        if dx * dx + dy * dy < min_motion * min_motion:
            cv2.circle(vis, start, 2, (0, 180, 0), -1, cv2.LINE_AA)
            continue
        end = (int(round(x1 + dx * arrow_scale)), int(round(y1 + dy * arrow_scale)))
        cv2.arrowedLine(vis, start, end, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(vis, start, 2, (0, 0, 255), -1, cv2.LINE_AA)

    put_label(vis, f"Tracked points: {len(new_pts)}", org=(8, 22))
    put_label(vis, f"Optical flow: MOTION BETWEEN FRAMES (arrows x{arrow_scale:g})",
              org=(8, vis.shape[0] - 10), scale=0.5, color=(0, 255, 0))
    return vis
