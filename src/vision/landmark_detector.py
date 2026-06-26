# =============================================================================
# src/vision/landmark_detector.py — MediaPipe Face Mesh wrapper
#
# MediaPipe Face Mesh detects 468 3-D landmarks on a human face in real time.
# This module wraps it so every other module receives a clean, typed result
# rather than raw MediaPipe proto objects.
#
# Landmark index reference (the ones we use):
#   Left eye  : 362, 385, 387, 263, 373, 380
#   Right eye : 33,  160, 158, 133, 153, 144
#   Mouth     : 61,  291, 39,  181, 0,   17,  269, 405
#   Nose tip  : 1
#   Head pose keypoints: 1 (nose), 152 (chin), 226/446 (eye corners),
#                        57/287 (mouth corners)
# =============================================================================

from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE


# ---------------------------------------------------------------------------
# Landmark index constants
# Using named constants makes the feature-extraction code self-documenting.
# ---------------------------------------------------------------------------

# Each eye is defined by 6 landmarks arranged around the eye contour.
# The ordering follows the EAR formula by Soukupová & Čech (2016).
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# Mouth landmarks: 4 vertical pairs + 2 horizontal corners
MOUTH_IDX = [61, 39, 0, 269, 291, 405, 17, 181]

# 6 stable 3-D points used for head-pose estimation via PnP
# These correspond to anatomically well-defined, non-deforming regions.
HEAD_POSE_IDX = [1, 152, 226, 446, 57, 287]


@dataclass
class LandmarkResult:
    """
    All outputs produced from one frame.

    Attributes
    ----------
    landmarks_px   : (468, 2) float array — landmark (x, y) in pixel coords
    landmarks_3d   : (468, 3) float array — landmark (x, y, z) normalised
    left_eye_pts   : (6, 2)  float array
    right_eye_pts  : (6, 2)  float array
    mouth_pts      : (8, 2)  float array
    head_pose_pts  : (6, 2)  float array — 2-D projections for PnP
    face_detected  : bool
    """
    landmarks_px:  np.ndarray = field(default_factory=lambda: np.zeros((468, 2)))
    landmarks_3d:  np.ndarray = field(default_factory=lambda: np.zeros((468, 3)))
    left_eye_pts:  np.ndarray = field(default_factory=lambda: np.zeros((6, 2)))
    right_eye_pts: np.ndarray = field(default_factory=lambda: np.zeros((6, 2)))
    mouth_pts:     np.ndarray = field(default_factory=lambda: np.zeros((8, 2)))
    head_pose_pts: np.ndarray = field(default_factory=lambda: np.zeros((6, 2)))
    face_detected: bool = False


class LandmarkDetector:
    """
    Detects 468 facial landmarks using MediaPipe Face Mesh.

    Lifecycle
    ---------
        detector = LandmarkDetector()
        detector.open()
        result = detector.process(bgr_frame)
        detector.close()

    Or use as a context manager:
        with LandmarkDetector() as det:
            result = det.process(frame)
    """

    def __init__(
        self,
        detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        tracking_confidence:  float = MIN_TRACKING_CONFIDENCE,
        max_faces:            int   = 1,
    ):
        self._detection_conf = detection_confidence
        self._tracking_conf  = tracking_confidence
        self._max_faces      = max_faces
        self._mesh           = None   # mediapipe FaceMesh instance

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Initialise the MediaPipe FaceMesh graph."""
        mp_face_mesh = mp.solutions.face_mesh
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,          # video mode — reuses tracking
            max_num_faces=self._max_faces,
            refine_landmarks=True,            # enables iris landmarks too
            min_detection_confidence=self._detection_conf,
            min_tracking_confidence=self._tracking_conf,
        )
        print("[INFO] LandmarkDetector initialised.")

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._mesh:
            self._mesh.close()
            print("[INFO] LandmarkDetector closed.")

    # ── Core API ──────────────────────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> LandmarkResult:
        """
        Run Face Mesh on one BGR frame.

        MediaPipe expects RGB input, so we convert internally.  The caller
        always receives pixel-space coordinates so downstream maths is
        straightforward.

        Parameters
        ----------
        bgr_frame : (H, W, 3) uint8 ndarray from cv2.VideoCapture

        Returns
        -------
        LandmarkResult — face_detected=False if no face found in frame
        """
        result = LandmarkResult()

        if self._mesh is None:
            raise RuntimeError("Call LandmarkDetector.open() before process().")

        h, w = bgr_frame.shape[:2]

        # MediaPipe requires RGB; conversion is cheap (~1 ms)
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Disable writeable flag — avoids an unnecessary copy inside MediaPipe
        rgb.flags.writeable = False
        mp_result = self._mesh.process(rgb)
        rgb.flags.writeable = True

        if not mp_result.multi_face_landmarks:
            return result   # face_detected stays False

        # Use only the first detected face (max_num_faces=1)
        face = mp_result.multi_face_landmarks[0]

        # Build arrays dynamically based on the number of landmarks returned
        num_landmarks = len(face.landmark)

        px = np.zeros((num_landmarks, 2), dtype=np.float32)
        nd = np.zeros((num_landmarks, 3), dtype=np.float32)

        for i, lm in enumerate(face.landmark):
            px[i] = (lm.x * w, lm.y * h)
            nd[i] = (lm.x, lm.y, lm.z)

        result.landmarks_px  = px
        result.landmarks_3d  = nd
        result.left_eye_pts  = px[LEFT_EYE_IDX]
        result.right_eye_pts = px[RIGHT_EYE_IDX]
        result.mouth_pts     = px[MOUTH_IDX]
        result.head_pose_pts = px[HEAD_POSE_IDX]
        result.face_detected = True

        return result

    def draw_landmarks(
        self,
        frame: np.ndarray,
        result: LandmarkResult,
        draw_tesselation: bool = False,
        dot_color: tuple = (0, 255, 0),
        dot_radius: int = 1,
    ) -> np.ndarray:
        """
        Draw detected landmarks onto a copy of the frame.

        Parameters
        ----------
        frame            : original BGR frame
        result           : output of process()
        draw_tesselation : draw the full 468-point mesh (slower but informative)
        dot_color        : BGR colour for landmark dots
        dot_radius       : radius in pixels

        Returns
        -------
        Annotated BGR frame (original is not modified)
        """
        annotated = frame.copy()
        if not result.face_detected:
            return annotated

        if draw_tesselation:
            # Draw every landmark as a tiny dot
            for x, y in result.landmarks_px.astype(int):
                cv2.circle(annotated, (x, y), dot_radius, dot_color, -1)
        else:
            # Draw only the feature landmarks (eyes, mouth) — much faster
            for pts, color in [
                (result.left_eye_pts,  (0, 255, 0)),
                (result.right_eye_pts, (0, 255, 0)),
                (result.mouth_pts,     (0, 200, 255)),
            ]:
                for x, y in pts.astype(int):
                    cv2.circle(annotated, (x, y), 2, color, -1)

        return annotated

    # ── Context-manager support ───────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# Smoke-test
# python -m src.vision.landmark_detector
# =============================================================================

if __name__ == "__main__":
    from src.vision.camera import Camera

    print("[TEST] Landmark detector smoke-test. Press 'q' to quit.")

    with Camera() as cam, LandmarkDetector() as detector:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            result = detector.process(frame)
            annotated = detector.draw_landmarks(frame, result, draw_tesselation=True)

            status = "Face DETECTED" if result.face_detected else "No face"
            cv2.putText(annotated, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if result.face_detected else (0, 0, 255), 2)

            cv2.imshow("Landmark Detector Test — q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
