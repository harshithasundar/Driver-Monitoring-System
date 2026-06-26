# =============================================================================
# src/features/extractor.py — Unified feature extraction pipeline
#
# This is the single entry point for the feature layer.  It takes a
# LandmarkResult and returns a FeatureVector — the structured row of numbers
# that the ML model will consume.
#
# Design decision: keeping all feature calls in one class means the data-
# collection script, the trainer, and the live predictor all share exactly
# the same feature computation code.  No drift between training and inference.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.features.ear import compute_avg_ear, is_eye_closed
from src.features.mar import compute_mar, is_yawning
from src.features.head_pose import compute_head_pose, is_head_drooping
from src.vision.landmark_detector import LandmarkResult
from config import FRAME_WIDTH, FRAME_HEIGHT


@dataclass
class FeatureVector:
    """
    One row of features derived from a single video frame.

    All fields are plain Python floats so they serialise cleanly to CSV
    without any numpy dtype surprises.
    """
    ear_left:  float = 0.0
    ear_right: float = 0.0
    ear_avg:   float = 0.0
    mar:       float = 0.0
    pitch:     float = 0.0
    yaw:       float = 0.0
    roll:      float = 0.0

    # Derived boolean flags — not used as model features, but useful for
    # display and rule-based checks.
    eye_closed:    bool = False
    is_yawning:    bool = False
    head_drooping: bool = False

    def to_model_input(self) -> np.ndarray:
        """
        Return a (1, 7) numpy array ready to pass to model.predict().
        Column order matches config.FEATURE_COLUMNS exactly.
        """
        return np.array([[
            self.ear_left,
            self.ear_right,
            self.ear_avg,
            self.mar,
            self.pitch,
            self.yaw,
            self.roll,
        ]], dtype=np.float32)

    def to_dict(self) -> dict:
        """Return feature values as a plain dict (used when writing CSV rows)."""
        return {
            "ear_left":  self.ear_left,
            "ear_right": self.ear_right,
            "ear_avg":   self.ear_avg,
            "mar":       self.mar,
            "pitch":     self.pitch,
            "yaw":       self.yaw,
            "roll":      self.roll,
        }


class FeatureExtractor:
    """
    Computes all features from a LandmarkResult in one call.

    Usage
    -----
        extractor = FeatureExtractor()
        fv = extractor.extract(landmark_result)
        print(fv.ear_avg, fv.mar, fv.pitch)
    """

    def __init__(
        self,
        frame_width:  int = FRAME_WIDTH,
        frame_height: int = FRAME_HEIGHT,
    ):
        self._fw = frame_width
        self._fh = frame_height

    def extract(self, result: LandmarkResult) -> FeatureVector:
        """
        Extract all features from one LandmarkResult.

        If no face was detected the method returns a zeroed FeatureVector
        rather than raising — the caller can check result.face_detected
        before calling this.

        Parameters
        ----------
        result : LandmarkResult from LandmarkDetector.process()

        Returns
        -------
        FeatureVector
        """
        fv = FeatureVector()

        if not result.face_detected:
            return fv

        # ── EAR ──────────────────────────────────────────────────────────────
        fv.ear_left, fv.ear_right, fv.ear_avg = compute_avg_ear(
            result.left_eye_pts,
            result.right_eye_pts,
        )
        fv.eye_closed = is_eye_closed(fv.ear_avg)

        # ── MAR ──────────────────────────────────────────────────────────────
        fv.mar       = compute_mar(result.mouth_pts)
        fv.is_yawning = is_yawning(fv.mar)

        # ── Head Pose ─────────────────────────────────────────────────────────
        fv.pitch, fv.yaw, fv.roll = compute_head_pose(
            result.head_pose_pts,
            self._fw,
            self._fh,
        )
        fv.head_drooping = is_head_drooping(fv.pitch, fv.roll)

        return fv
