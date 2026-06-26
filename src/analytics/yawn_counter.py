"""
yawn_counter.py

Counts complete yawns using MAR.
"""

from config import MAR_THRESHOLD, YAWN_MIN_FRAMES


class YawnCounter:

    def __init__(self, min_frames=YAWN_MIN_FRAMES):
        self.min_frames = min_frames

        self.open_frames = 0
        self.total_yawns = 0

        self.yawning = False

    def update(self, mar):

        if mar > MAR_THRESHOLD:

            self.open_frames += 1

            if self.open_frames >= self.min_frames:
                self.yawning = True

        else:

            if self.yawning:
                self.total_yawns += 1

            self.open_frames = 0
            self.yawning = False

        return self.total_yawns