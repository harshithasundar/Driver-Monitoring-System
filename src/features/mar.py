# =============================================================================
# src/features/mar.py — Mouth Aspect Ratio (MAR) computation
#
# MAR mirrors the EAR concept but applied to the mouth opening.
# A large mouth opening indicates a yawn — a strong drowsiness signal.
#
# We use 8 mouth landmarks arranged as:
#
#         p2 ── p3
#        /         \
#      p1           p4
#        \         /
#         p8 ── p7
#             │
#           p5, p6  (inner vertical)
#
# Using MediaPipe indices [61, 291, 39, 181, 0, 17, 269, 405]:
#   p1=61  (left corner)
#   p2=39  (upper-left)
#   p3=0   (upper-middle)
#   p4=269 (upper-right)
#   p5=291 (right corner)
#   p6=405 (lower-right)
#   p7=17  (lower-middle)
#   p8=181 (lower-left)
#
#   MAR = (‖p2-p8‖ + ‖p3-p7‖ + ‖p4-p6‖) / (2 × ‖p1-p5‖)
#
# When mouth is closed: MAR ≈ 0.2–0.4
# During a yawn:        MAR > 0.6
# =============================================================================

import numpy as np
from config import MAR_THRESHOLD


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_mar(mouth_pts: np.ndarray) -> float:
    """
    Compute the Mouth Aspect Ratio.

    Parameters
    ----------
    mouth_pts : (8, 2) float array
        Landmark coordinates ordered as:
        [p1(61), p2(39), p3(0), p4(269), p5(291), p6(405), p7(17), p8(181)]

    Returns
    -------
    float — MAR value

    Raises
    ------
    ValueError if mouth_pts shape is not (8, 2)
    """
    if mouth_pts.shape != (8, 2):
        raise ValueError(f"mouth_pts must be (8, 2), got {mouth_pts.shape}")

    # Three vertical distances (numerator)
    v1 = _euclidean(mouth_pts[1], mouth_pts[7])   # ‖p2 - p8‖
    v2 = _euclidean(mouth_pts[2], mouth_pts[6])   # ‖p3 - p7‖
    v3 = _euclidean(mouth_pts[3], mouth_pts[5])   # ‖p4 - p6‖

    # Horizontal distance (denominator)
    h  = _euclidean(mouth_pts[0], mouth_pts[4])   # ‖p1 - p5‖

    if h < 1e-6:
        return 0.0

    return (v1 + v2 + v3) / (2.0 * h)


def is_yawning(mar: float, threshold: float = MAR_THRESHOLD) -> bool:
    """Return True if MAR exceeds the yawn threshold."""
    return mar > threshold
