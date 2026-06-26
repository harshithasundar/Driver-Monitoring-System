# =============================================================================
# tests/test_features.py — Unit tests for feature computation modules
#
# Run with:  python -m pytest tests/ -v
# =============================================================================

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.ear import compute_ear, compute_avg_ear, is_eye_closed
from src.features.mar import compute_mar, is_yawning
from src.features.head_pose import compute_head_pose, is_head_drooping


# =============================================================================
# EAR tests
# =============================================================================

class TestEAR:
    """Eye Aspect Ratio tests."""

    def _open_eye(self) -> np.ndarray:
        """
        Synthetic open-eye landmarks — EAR ≈ 0.30.

        Eye width (horizontal) = 40 px, vertical opening = 12 px.
        EAR = (12 + 12) / (2 * 40) = 0.30  ✓
        """
        return np.array([
            [0,   0],   # p1 — left corner
            [10,  6],   # p2 — upper-left
            [30,  6],   # p3 — upper-right
            [40,  0],   # p4 — right corner
            [30, -6],   # p5 — lower-right
            [10, -6],   # p6 — lower-left
        ], dtype=np.float32)

    def _closed_eye(self) -> np.ndarray:
        """Synthetic closed-eye landmarks — EAR ≈ 0."""
        return np.array([
            [0, 0],
            [1, 0.1],
            [3, 0.1],
            [4, 0],
            [3, -0.1],
            [1, -0.1],
        ], dtype=np.float32)

    def test_open_eye_ear_is_reasonable(self):
        ear = compute_ear(self._open_eye())
        assert 0.25 < ear < 0.60, f"Expected open EAR ~ 0.30, got {ear:.4f}"

    def test_closed_eye_ear_is_low(self):
        ear = compute_ear(self._closed_eye())
        assert ear < 0.10, f"Expected closed EAR < 0.10, got {ear:.4f}"

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            compute_ear(np.zeros((5, 2)))

    def test_avg_ear_returns_three_values(self):
        eye = self._open_eye()
        el, er, avg = compute_avg_ear(eye, eye)
        assert el == er
        assert avg == el

    def test_is_eye_closed_true(self):
        ear = compute_ear(self._closed_eye())
        assert is_eye_closed(ear, threshold=0.25)

    def test_is_eye_closed_false(self):
        ear = compute_ear(self._open_eye())
        assert not is_eye_closed(ear, threshold=0.25)


# =============================================================================
# MAR tests
# =============================================================================

class TestMAR:
    """Mouth Aspect Ratio tests."""

    def _closed_mouth(self) -> np.ndarray:
        """Flat mouth — low MAR."""
        return np.array([
            [0,  0],   # p1  61 — left corner
            [1,  0.2], # p2  39
            [3,  0.2], # p3  0
            [5,  0.2], # p4  269
            [6,  0],   # p5  291 — right corner
            [5, -0.2], # p6  405
            [3, -0.2], # p7  17
            [1, -0.2], # p8  181
        ], dtype=np.float32)

    def _open_mouth(self) -> np.ndarray:
        """Wide-open mouth — high MAR."""
        return np.array([
            [0,  0],
            [1,  3],
            [3,  4],
            [5,  3],
            [6,  0],
            [5, -3],
            [3, -4],
            [1, -3],
        ], dtype=np.float32)

    def test_closed_mouth_low_mar(self):
        mar = compute_mar(self._closed_mouth())
        assert mar < 0.20, f"Closed mouth MAR should be < 0.20, got {mar:.4f}"

    def test_open_mouth_high_mar(self):
        mar = compute_mar(self._open_mouth())
        assert mar > 0.50, f"Open mouth MAR should be > 0.50, got {mar:.4f}"

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            compute_mar(np.zeros((6, 2)))

    def test_is_yawning_true(self):
        mar = compute_mar(self._open_mouth())
        assert is_yawning(mar, threshold=0.5)

    def test_is_yawning_false(self):
        mar = compute_mar(self._closed_mouth())
        assert not is_yawning(mar, threshold=0.5)


# =============================================================================
# Head pose tests
# =============================================================================

class TestHeadPose:

    def test_returns_three_floats(self):
        # Plausible 2-D image points for a forward-facing head at 640x480
        pts = np.array([
            [320, 240],  # nose tip
            [320, 360],  # chin
            [200, 200],  # left eye corner
            [440, 200],  # right eye corner
            [260, 300],  # left mouth
            [380, 300],  # right mouth
        ], dtype=np.float32)
        pitch, yaw, roll = compute_head_pose(pts, 640, 480)
        assert isinstance(pitch, float)
        assert isinstance(yaw,   float)
        assert isinstance(roll,  float)

    def test_wrong_shape_returns_zeros(self):
        pitch, yaw, roll = compute_head_pose(np.zeros((4, 2)), 640, 480)
        assert (pitch, yaw, roll) == (0.0, 0.0, 0.0)

    def test_drooping_head(self):
        assert is_head_drooping(pitch=20.0, roll=0.0, pitch_threshold=15.0)

    def test_normal_head(self):
        assert not is_head_drooping(pitch=5.0, roll=3.0,
                                    pitch_threshold=15.0, roll_threshold=20.0)
