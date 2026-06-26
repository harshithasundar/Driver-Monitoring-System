"""
fatigue_score.py

Computes a fatigue score using the rolling average
of drowsiness probabilities.
"""

from src.fatigue.rolling_buffer import RollingBuffer
from src.fatigue.alert_levels import get_alert_level


class FatigueScorer:

    def __init__(self, window_size: int = 30):
        self.buffer = RollingBuffer(window_size)

    def update(self, probability: float):
        """
        Add a new drowsiness probability.

        Returns:
            score (0-100)
            alert level
        """

        self.buffer.add(probability)

        avg_probability = self.buffer.average()

        score = avg_probability * 100

        level = get_alert_level(score)

        return score, level

    def reset(self):
        self.buffer.clear()