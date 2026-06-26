"""
alarm.py

Simple alarm utility for Windows.
"""

import winsound


def play_alarm():
    """
    Play a short warning beep.
    """
    frequency = 1500
    duration = 600

    winsound.Beep(frequency, duration)