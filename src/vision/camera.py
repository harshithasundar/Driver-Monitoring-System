# =============================================================================
# src/vision/camera.py — Webcam capture abstraction
#
# Wraps OpenCV's VideoCapture in a class so the rest of the project never
# touches raw cv2 capture calls directly.  This makes it trivial to swap
# in a video file or an IP-camera stream later without touching any other
# module.
# =============================================================================

import cv2
import sys
import time
from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS_TARGET,
)


class Camera:
    """
    Manages webcam lifecycle: open → read frames → release.

    Usage
    -----
        cam = Camera()
        cam.open()
        ret, frame = cam.read()
        cam.release()

    Or use as a context manager:
        with Camera() as cam:
            ret, frame = cam.read()
    """

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = FPS_TARGET,
    ):
        self.index  = index
        self.width  = width
        self.height = height
        self.fps    = fps
        self._cap   = None          # cv2.VideoCapture object
        self._frame_count   = 0
        self._start_time    = None

    # ── Public API ────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open the camera device and configure resolution + FPS.
        Exits the program with a clear message if the camera cannot be opened
        (e.g. another process owns it, or no webcam is attached).
        """
        self._cap = cv2.VideoCapture(self.index)

        if not self._cap.isOpened():
            print(
                f"[ERROR] Cannot open camera at index {self.index}. "
                "Check that no other application is using it."
            )
            sys.exit(1)

        # Request resolution and FPS — the driver may not honour every value,
        # but we set them explicitly so behaviour is predictable.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        self._start_time = time.time()
        self._frame_count = 0

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Camera opened — resolution: {actual_w}×{actual_h}")

    def read(self):
        """
        Capture one frame.

        Returns
        -------
        ret   : bool   — True if a frame was successfully grabbed
        frame : ndarray or None — BGR image (H, W, 3)
        """
        if self._cap is None or not self._cap.isOpened():
            print("[ERROR] Camera.read() called before Camera.open().")
            return False, None

        ret, frame = self._cap.read()
        if ret:
            self._frame_count += 1
        return ret, frame

    def release(self) -> None:
        """Release the camera resource and print session statistics."""
        if self._cap and self._cap.isOpened():
            elapsed = time.time() - self._start_time if self._start_time else 0
            avg_fps = self._frame_count / elapsed if elapsed > 0 else 0
            print(
                f"[INFO] Camera released — "
                f"{self._frame_count} frames in {elapsed:.1f}s "
                f"(avg {avg_fps:.1f} fps)"
            )
            self._cap.release()

    def is_opened(self) -> bool:
        """Return True if the camera device is currently open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_count(self) -> int:
        """Total frames captured since open()."""
        return self._frame_count

    # ── Context-manager support ───────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        cv2.destroyAllWindows()
        # Returning False lets any exception propagate normally
        return False


# =============================================================================
# Quick smoke-test — run this file directly to verify your webcam works.
# python -m src.vision.camera
# Press 'q' to quit.
# =============================================================================

if __name__ == "__main__":
    print("[TEST] Running Camera smoke-test. Press 'q' to quit.")

    with Camera() as cam:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("[WARN] Empty frame received — retrying…")
                continue

            # Overlay frame counter so we can see it's truly live
            cv2.putText(
                frame,
                f"Frame: {cam.frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Test — press q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    print("[TEST] Camera test complete.")
