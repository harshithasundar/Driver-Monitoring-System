# =============================================================================
# src/features/head_pose.py — Head Pose Estimation (pitch, yaw, roll)
#
# Uses OpenCV's solvePnP to recover the 3-D rotation of the head from 6
# 2-D landmark points projected onto the image plane.
#
# solvePnP solves the Perspective-n-Point problem:
#   Given N known 3-D points and their 2-D image projections, find the
#   rotation (R) and translation (T) that maps 3-D → 2-D.
#
# We then convert R into Euler angles:
#   pitch — nodding  (head tilts forward/backward)
#   yaw   — shaking  (head turns left/right)
#   roll  — tilting  (head tilts ear-to-shoulder)
#
# Drowsiness signatures:
#   |pitch| > 15°  → head drooping forward
#   |roll|  > 20°  → head falling sideways
# =============================================================================

import cv2
import numpy as np
from typing import Tuple
from config import FRAME_WIDTH, FRAME_HEIGHT, HEAD_PITCH_THRESHOLD, HEAD_ROLL_THRESHOLD


# ---------------------------------------------------------------------------
# 3-D reference model of a generic human head (in millimetres).
# These are canonical coordinates for the 6 landmarks we use.
# They do NOT change per-person — we assume an average head shape.
# Source: standard face model used widely in academic head-pose literature.
# ---------------------------------------------------------------------------
_MODEL_3D_POINTS = np.array([
    [0.0,      0.0,      0.0  ],   # 1   — nose tip
    [0.0,     -63.6,    -12.5 ],   # 152 — chin
    [-43.3,    32.7,    -26.0 ],   # 226 — left eye, left corner
    [43.3,     32.7,    -26.0 ],   # 446 — right eye, right corner
    [-28.9,   -28.9,    -24.1 ],   # 57  — left mouth corner
    [28.9,    -28.9,    -24.1 ],   # 287 — right mouth corner
], dtype=np.float64)


def _build_camera_matrix(width: int, height: int) -> np.ndarray:
    """
    Approximate the camera intrinsic matrix.

    For a real deployment you would calibrate the camera once and load
    the matrix.  Here we use the common approximation:
        focal_length ≈ image_width
        principal_point ≈ image_centre
    This is accurate enough for drowsiness detection.
    """
    focal_length = width
    cx, cy = width / 2.0, height / 2.0
    return np.array([
        [focal_length, 0,            cx],
        [0,            focal_length, cy],
        [0,            0,            1 ],
    ], dtype=np.float64)


# Lens distortion — assumed zero (standard webcam approximation)
_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


def compute_head_pose(
    image_pts: np.ndarray,
    frame_width:  int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
) -> Tuple[float, float, float]:
    """
    Estimate head orientation (pitch, yaw, roll) in degrees.

    Parameters
    ----------
    image_pts    : (6, 2) float array — 2-D pixel positions of the 6
                   landmarks in HEAD_POSE_IDX order
    frame_width  : image width in pixels
    frame_height : image height in pixels

    Returns
    -------
    (pitch, yaw, roll) in degrees — (0, 0, 0) on failure
    """
    if image_pts.shape != (6, 2):
        return 0.0, 0.0, 0.0

    camera_matrix = _build_camera_matrix(frame_width, frame_height)

    success, rotation_vec, _ = cv2.solvePnP(
        _MODEL_3D_POINTS,
        image_pts.astype(np.float64),
        camera_matrix,
        _DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Convert rotation vector → rotation matrix (Rodrigues formula)
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)

    # Decompose rotation matrix into Euler angles via RQDecomp3x3
    # Returns angles in degrees when we multiply by 180/π
    euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)

    pitch = euler_angles[0]  # X-axis — nod
    yaw   = euler_angles[1]  # Y-axis — shake
    roll  = euler_angles[2]  # Z-axis — tilt

    # Normalize pitch to [-90, 90]
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return float(pitch), float(yaw), float(roll)


def is_head_drooping(
    pitch: float,
    roll:  float,
    pitch_threshold: float = HEAD_PITCH_THRESHOLD,
    roll_threshold:  float = HEAD_ROLL_THRESHOLD,
) -> bool:
    """
    Return True if the head pose indicates drowsiness-related drooping.

    Parameters
    ----------
    pitch           : pitch angle in degrees (positive = head down)
    roll            : roll angle in degrees
    pitch_threshold : maximum acceptable pitch
    roll_threshold  : maximum acceptable roll
    """
    return abs(pitch) > pitch_threshold or abs(roll) > roll_threshold


def draw_pose_axes(
    frame: np.ndarray,
    image_pts: np.ndarray,
    pitch: float,
    yaw:   float,
    roll:  float,
) -> np.ndarray:
    """
    Overlay pitch/yaw/roll text and a nose-direction arrow on the frame.

    Returns a copy of the frame with annotations drawn.
    """
    annotated = frame.copy()

    # Text overlay
    h = frame.shape[0]
    cv2.putText(annotated, f"Pitch: {pitch:+.1f}", (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
    cv2.putText(annotated, f"Yaw  : {yaw:+.1f}",  (10, h - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
    cv2.putText(annotated, f"Roll : {roll:+.1f}",  (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    return annotated
