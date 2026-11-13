"""Image processing: spatial gradients, feature detection, and sparse optical flow.

Two different ideas live here:
  * Gradients (Ix, Iy) describe how brightness changes ACROSS SPACE in ONE frame.
  * Optical flow estimates how points MOVE BETWEEN TWO frames over time.
Lucas-Kanade actually relies on gradients: a point can only be tracked where
the image has strong gradients in more than one direction (a "corner").
"""

from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Spatial gradients
# --------------------------------------------------------------------------
def compute_gradients(gray, ksize=3):
    """Return (Ix, Iy, magnitude) as float32 arrays.

    Sobel approximates the derivative of image brightness:
      Ix > 0 where the image gets brighter going right, < 0 where it gets darker.
      Iy > 0 where the image gets brighter going down,  < 0 where it gets darker.
    We keep float32 so negative values are preserved (uint8 would clip them).
    """
    ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(ix, iy)  # sqrt(Ix^2 + Iy^2): edge strength
    return ix, iy, magnitude


# --------------------------------------------------------------------------
# Sparse optical flow (Shi-Tomasi + pyramidal Lucas-Kanade)
# --------------------------------------------------------------------------
@dataclass
class TrackerParams:
    # Shi-Tomasi corner detection
    max_corners: int = 200        # upper bound on tracked points
    quality_level: float = 0.01   # min corner strength relative to the best corner
    min_distance: int = 8         # min pixel spacing between corners
    block_size: int = 7           # neighbourhood used to evaluate "cornerness"

    # Pyramidal Lucas-Kanade
    win_size: int = 21            # search window (pixels) at each pyramid level
    max_level: int = 3            # pyramid levels; more levels = handles larger motion
    lk_iterations: int = 30
    lk_epsilon: float = 0.01

    # Track quality control
    min_points: int = 60          # replenish features when fewer remain
    fb_threshold: float = 1.0     # max forward-backward error (pixels) to accept a track


class FlowTracker:
    """Tracks Shi-Tomasi corners between consecutive grayscale frames."""

    def __init__(self, params=None):
        self.params = params or TrackerParams()
        p = self.params
        self._lk_args = dict(
            winSize=(p.win_size, p.win_size),
            maxLevel=p.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      p.lk_iterations, p.lk_epsilon),
        )
        self.reset()

    def reset(self):
        """Forget the previous frame and all tracked points."""
        self.prev_gray = None
        self.points = None  # shape (N, 1, 2), float32

    def _detect(self, gray, existing=None):
        """Find new Shi-Tomasi corners, avoiding areas already covered by points."""
        p = self.params
        mask = None
        if existing is not None and len(existing) > 0:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            for x, y in existing.reshape(-1, 2):
                cv2.circle(mask, (int(x), int(y)), p.min_distance, 0, -1)

        want = p.max_corners - (0 if existing is None else len(existing))
        if want <= 0:
            return existing

        # Shi-Tomasi picks pixels where the gradient structure matrix has two
        # large eigenvalues -> texture in two directions -> trackable.
        new = cv2.goodFeaturesToTrack(
            gray, maxCorners=want, qualityLevel=p.quality_level,
            minDistance=p.min_distance, blockSize=p.block_size, mask=mask)

        if new is None:
            return existing if existing is not None else np.empty((0, 1, 2), np.float32)
        if existing is None or len(existing) == 0:
            return new.astype(np.float32)
        return np.vstack([existing, new]).astype(np.float32)

    def update(self, gray):
        """Track points from the previous frame into `gray`.

        Returns (old_pts, new_pts), each an (N, 2) float32 array of valid tracks.
        """
        empty = np.empty((0, 2), np.float32)

        # First frame (or after reset): nothing to compare against yet.
        if self.prev_gray is None or self.points is None or len(self.points) == 0:
            self.points = self._detect(gray)
            self.prev_gray = gray
            return empty, empty

        p0 = self.points

        # Forward pass: where did each point go from prev -> current?
        p1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self._lk_args)
        # Backward pass: track the result back current -> prev.
        p0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self._lk_args)

        # Forward-backward check: a reliable track returns close to where it started.
        fb_error = np.linalg.norm((p0 - p0_back).reshape(-1, 2), axis=1)

        h, w = gray.shape
        xy = p1.reshape(-1, 2)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)

        good = (
            (st_fwd.ravel() == 1)
            & (st_bwd.ravel() == 1)
            & (fb_error < self.params.fb_threshold)
            & inside
        )

        old_pts = p0.reshape(-1, 2)[good]
        new_pts = xy[good]

        # Keep surviving points; top up with fresh corners if too few remain.
        self.points = new_pts.reshape(-1, 1, 2).astype(np.float32)
        if len(self.points) < self.params.min_points:
            self.points = self._detect(gray, existing=self.points)

        self.prev_gray = gray
        return old_pts, new_pts
