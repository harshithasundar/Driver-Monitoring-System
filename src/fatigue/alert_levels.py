"""
alert_levels.py

Maps fatigue scores to alert levels.
"""

from enum import Enum


class AlertLevel(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    DROWSY = "DROWSY"
    CRITICAL = "CRITICAL"


def get_alert_level(score: float) -> AlertLevel:
    """
    Convert fatigue score (0-100) into an alert level.
    """

    if score <= 30:
        return AlertLevel.NORMAL

    elif score <= 60:
        return AlertLevel.WARNING

    elif score <= 80:
        return AlertLevel.DROWSY

    else:
        return AlertLevel.CRITICAL