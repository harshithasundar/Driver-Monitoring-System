import time

from src.alerts.alert_manager import AlertManager

manager = AlertManager()

while True:

    triggered = manager.update(90)

    if triggered:
        print("Alarm!")

    time.sleep(1)