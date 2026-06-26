"""
blink_counter.py

Counts complete eye blinks using EAR.
"""

from config import EAR_THRESHOLD, BLINK_MIN_FRAMES


class BlinkCounter:

    def __init__(self, min_frames=BLINK_MIN_FRAMES):
        self.min_frames = min_frames

        self.closed_frames = 0
        self.total_blinks = 0

        self.eye_closed = False

    def update(self, ear):

        if ear < EAR_THRESHOLD:

            self.closed_frames += 1

            if self.closed_frames >= self.min_frames:
                self.eye_closed = True

        else:

            if self.eye_closed:

                self.total_blinks += 1

            self.closed_frames = 0
            self.eye_closed = False

        return self.total_blinks