"""
session_stats.py

Tracks statistics for the current driving session.
"""

import time


class SessionStats:

    def __init__(self):

        self.start_time = time.time()

        self.max_fatigue = 0.0

        self.total_fatigue = 0.0
        self.frames = 0

        self.total_alerts = 0

        self.total_blinks = 0
        self.total_yawns = 0

    def update_fatigue(self, score):

        self.frames += 1

        self.total_fatigue += score

        self.max_fatigue = max(
            self.max_fatigue,
            score,
        )

    def set_blinks(self, count):

        self.total_blinks = count

    def set_yawns(self, count):

        self.total_yawns = count

    def increment_alerts(self):

        self.total_alerts += 1

    @property
    def average_fatigue(self):

        if self.frames == 0:
            return 0

        return self.total_fatigue / self.frames

    @property
    def session_duration(self):

        seconds = int(time.time() - self.start_time)

        minutes = seconds // 60

        seconds = seconds % 60

        return f"{minutes:02}:{seconds:02}"

    def summary(self):

        return {
            "Duration": self.session_duration,
            "Average Fatigue": round(self.average_fatigue, 1),
            "Maximum Fatigue": round(self.max_fatigue, 1),
            "Blinks": self.total_blinks,
            "Yawns": self.total_yawns,
            "Alerts": self.total_alerts,
        }