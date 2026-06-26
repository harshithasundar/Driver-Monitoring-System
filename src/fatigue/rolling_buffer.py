"""
rolling_buffer.py

Maintains a fixed-size rolling buffer of recent drowsiness probabilities.
"""

from collections import deque
from typing import Deque


class RollingBuffer:
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.buffer: Deque[float] = deque(maxlen=max_size)

    def add(self, probability: float) -> None:
        """
        Add a new drowsiness probability.

        probability must be between 0.0 and 1.0
        """
        probability = max(0.0, min(1.0, probability))
        self.buffer.append(probability)

    def average(self) -> float:
        """
        Returns the average probability.
        """
        if len(self.buffer) == 0:
            return 0.0

        return sum(self.buffer) / len(self.buffer)

    def clear(self) -> None:
        """
        Clears the buffer.
        """
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)