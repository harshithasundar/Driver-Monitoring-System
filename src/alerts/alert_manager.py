"""
alert_manager.py

Triggers alarms with cooldown.
"""

import time

from src.alerts.alarm import play_alarm


class AlertManager:

    def __init__(
        self,
        threshold=80,
        trigger_time=3,
        cooldown=5,
    ):

        self.threshold = threshold
        self.trigger_time = trigger_time
        self.cooldown = cooldown

        self.start_time = None
        self.last_alarm = 0

    def update(self, fatigue_score):

        now = time.time()

        if fatigue_score >= self.threshold:

            if self.start_time is None:
                self.start_time = now

            duration = now - self.start_time

            if duration >= self.trigger_time:

                if now - self.last_alarm >= self.cooldown:

                    play_alarm()

                    self.last_alarm = now

                    return True

        else:

            self.start_time = None

        return False