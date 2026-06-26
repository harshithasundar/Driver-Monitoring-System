# =============================================================================
# src/features/ear.py — Eye Aspect Ratio (EAR) computation
#
# Formula from: Soukupová & Čech, "Real-Time Eye Blink Detection using
# Facial Landmarks", CVWW 2016.
#
# For each eye, 6 landmark points are used:
#
#        p2 ─── p3
#       /           \
#     p1             p4
#       \           /
#        p6 ─── p5
#
#   EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2 × ‖p1-p4‖)
#
# When the eye is open,  EAR ≈ 0.25–0.35
# When the eye is closed, EAR → 0
# =============================================================================

import numpy as np
from config import EAR_THRESHOLD


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(a - b))


def compute_ear(eye_pts: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio for one eye.

    Parameters
    ----------
    eye_pts : (6, 2) float array
        Landmark coordinates in pixel space, ordered as:
        [p1, p2, p3, p4, p5, p6]  (see diagram above)

    Returns
    -------
    float — EAR value (0.0 when closed, ~0.3 when open)

    Raises
    ------
    ValueError if eye_pts does not have shape (6, 2)
    """
    if eye_pts.shape != (6, 2):
        raise ValueError(f"eye_pts must be (6, 2), got {eye_pts.shape}")

    # Vertical distances (numerator)
    v1 = _euclidean(eye_pts[1], eye_pts[5])   # ‖p2 - p6‖
    v2 = _euclidean(eye_pts[2], eye_pts[4])   # ‖p3 - p5‖

    # Horizontal distance (denominator)
    h  = _euclidean(eye_pts[0], eye_pts[3])   # ‖p1 - p4‖

    # Guard against division by zero (can happen with extreme head angles)
    if h < 1e-6:
        return 0.0

    return (v1 + v2) / (2.0 * h)


def compute_avg_ear(
    left_eye_pts: np.ndarray,
    right_eye_pts: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute EAR for both eyes and their average.

    Parameters
    ----------
    left_eye_pts  : (6, 2) array
    right_eye_pts : (6, 2) array

    Returns
    -------
    (ear_left, ear_right, ear_avg) — all floats
    """
    ear_left  = compute_ear(left_eye_pts)
    ear_right = compute_ear(right_eye_pts)
    ear_avg   = (ear_left + ear_right) / 2.0
    return ear_left, ear_right, ear_avg


def is_eye_closed(ear_avg: float, threshold: float = EAR_THRESHOLD) -> bool:
    """Return True if average EAR is below the closure threshold."""
    return ear_avg < threshold
