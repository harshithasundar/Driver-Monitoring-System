"""
hud.py

Professional OpenCV dashboard for the Driver Drowsiness Detection System.
"""

import cv2


class HUD:

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw(
        self,
        frame,
        ear,
        mar,
        pitch,
        prediction,
        confidence,
        fatigue_score,
        alert_level,
        fps,
        blinks,
        yawns,
    ):

        h, w = frame.shape[:2]

        panel_width = 320

        cv2.rectangle(
            frame,
            (0, 0),
            (panel_width, h),
            (35, 35, 35),
            -1,
        )

        y = 35

        cv2.putText(
            frame,
            "Driver Monitoring System",
            (15, y),
            self.font,
            0.65,
            (255, 255, 255),
            2,
        )

        y += 40

        cv2.putText(frame, f"EAR : {ear:.3f}", (15, y),
                    self.font, 0.6, (255,255,255),1)

        y += 30

        cv2.putText(frame, f"MAR : {mar:.3f}", (15, y),
                    self.font, 0.6, (255,255,255),1)

        y += 30

        cv2.putText(frame, f"Pitch : {pitch:.1f}",
                    (15, y),
                    self.font,
                    0.6,
                    (255,255,255),
                    1)

        y += 45

        cv2.putText(frame,
                    f"Driver State : {prediction}",
                    (15,y),
                    self.font,
                    0.6,
                    (0,255,255),
                    2)

        y += 30

        cv2.putText(frame,
                    f"Confidence : {confidence:.1f}%",
                    (15,y),
                    self.font,
                    0.6,
                    (0,255,255),
                    2)

        y += 45

        cv2.putText(frame,
                    f"Fatigue : {fatigue_score:.0f}",
                    (15,y),
                    self.font,
                    0.7,
                    (255,255,255),
                    2)

        # Progress Bar

        y += 20

        bar_x = 15
        bar_y = y

        bar_width = 250
        bar_height = 22

        cv2.rectangle(
            frame,
            (bar_x,bar_y),
            (bar_x+bar_width,bar_y+bar_height),
            (80,80,80),
            2,
        )

        filled = int((fatigue_score/100)*bar_width)

        color=(0,255,0)

        if fatigue_score>30:
            color=(0,255,255)

        if fatigue_score>60:
            color=(0,165,255)

        if fatigue_score>80:
            color=(0,0,255)

        cv2.rectangle(
            frame,
            (bar_x,bar_y),
            (bar_x+filled,bar_y+bar_height),
            color,
            -1,
        )

        y += 50

        cv2.putText(frame,
                    alert_level.value,
                    (15,y),
                    self.font,
                    0.8,
                    color,
                    2)

        y += 40


        cv2.putText(
            frame,
            f"Blinks : {blinks}",
            (15, y),
            self.font,
            0.6,
            (255,255,255),
            1,
        )

        y += 30

        cv2.putText(
            frame,
            f"Yawns  : {yawns}",
            (15, y),
            self.font,
            0.6,
            (255,255,255),
            1,
        )

        y += 40

        cv2.putText(frame,
                    f"FPS : {fps:.1f}",
                    (15,y),
                    self.font,
                    0.6,
                    (200,200,200),
                    1)

        return frame