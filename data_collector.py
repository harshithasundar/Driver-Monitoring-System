# =============================================================================
# data_collector.py — Labeled training data collection from webcam
#
# Records feature vectors (EAR, MAR, head pose) frame-by-frame, attaches
# a label, and appends rows to data/processed/features.csv.
#
# Usage
# -----
#   python data_collector.py --label awake  --samples 600
#   python data_collector.py --label drowsy --samples 600
#
# The script runs interactively:
#   - A 3-second countdown lets you get into position.
#   - Only frames where a face is detected are recorded.
#   - Press 'q' at any time to stop early (partial saves are kept).
# =============================================================================

import sys
import os
import argparse
import time
import csv

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vision.camera import Camera
from src.vision.landmark_detector import LandmarkDetector
from src.features.extractor import FeatureExtractor
from config import (
    DATASET_PATH, PROC_DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN,
    LABEL_AWAKE, LABEL_DROWSY, LABEL_NAMES,
    COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_WHITE,
    FRAME_WIDTH, FRAME_HEIGHT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect labeled drowsiness feature data from webcam."
    )
    parser.add_argument("--label",    required=True, choices=["awake", "drowsy"])
    parser.add_argument("--samples",  type=int, default=600)
    parser.add_argument("--countdown",type=int, default=3)
    return parser.parse_args()


def _ensure_csv_header() -> None:
    """Create the CSV file with header row if it does not already exist."""
    os.makedirs(PROC_DATA_DIR, exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS + [TARGET_COLUMN])
            writer.writeheader()
        print(f"[INFO] Created new dataset at: {DATASET_PATH}")
    else:
        print(f"[INFO] Appending to existing dataset: {DATASET_PATH}")


def _append_row(row: dict) -> None:
    with open(DATASET_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS + [TARGET_COLUMN])
        writer.writerow(row)


def _draw_hud(frame, label_name, label_int, recorded, target, ear_avg, mar, pitch):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    pill_color = COLOR_GREEN if label_int == LABEL_AWAKE else COLOR_RED
    cv2.rectangle(frame, (10, 8), (160, 42), pill_color, -1)
    cv2.putText(frame, f"LABEL: {label_name}", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    pct = recorded / target if target > 0 else 0
    bar_w = w - 20
    cv2.rectangle(frame, (10, 50), (10 + bar_w, 70), (60, 60, 60), -1)
    cv2.rectangle(frame, (10, 50), (10 + int(bar_w * pct), 70), pill_color, -1)
    cv2.putText(frame, f"{recorded}/{target}  ({pct*100:.0f}%)",
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    cv2.putText(frame, f"EAR: {ear_avg:.3f}", (w - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)
    cv2.putText(frame, f"MAR: {mar:.3f}",     (w - 180, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)
    cv2.putText(frame, f"Pitch: {pitch:+.1f}",(w - 180, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1)
    cv2.putText(frame, "Press 'q' to stop early",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def _countdown(cam: Camera, seconds: int, label_name: str) -> None:
    start = time.time()
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        remaining = seconds - int(time.time() - start)
        if remaining <= 0:
            break
        cv2.putText(frame, f"Get ready ({label_name})... {remaining}",
                    (60, FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YELLOW, 3)
        cv2.imshow("Data Collector", frame)
        cv2.waitKey(1)


def collect(label_name: str, target_samples: int, countdown_sec: int) -> None:
    label_int = LABEL_AWAKE if label_name == "awake" else LABEL_DROWSY
    _ensure_csv_header()
    recorded = 0

    with Camera() as cam, LandmarkDetector() as detector:
        extractor = FeatureExtractor()
        print(f"[INFO] Starting {countdown_sec}s countdown…")
        _countdown(cam, countdown_sec, label_name.upper())
        print(f"[INFO] Recording {target_samples} '{label_name}' samples…")

        while recorded < target_samples:
            ret, frame = cam.read()
            if not ret:
                continue

            result    = detector.process(frame)
            annotated = detector.draw_landmarks(frame, result)

            if result.face_detected:
                fv  = extractor.extract(result)
                row = fv.to_dict()
                row[TARGET_COLUMN] = label_int
                _append_row(row)
                recorded += 1
                _draw_hud(annotated, label_name.upper(), label_int,
                          recorded, target_samples,
                          fv.ear_avg, fv.mar, fv.pitch)
            else:
                cv2.putText(annotated, "No face detected — move into frame",
                            (30, FRAME_HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)

            if recorded % 50 == 0 and recorded > 0:
                bar = "#" * (recorded // 50)
                print(f"  [{bar:<{target_samples // 50}}] {recorded}/{target_samples}")

            cv2.imshow("Data Collector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Early stop requested.")
                break

    print(f"\n[DONE] Recorded {recorded} '{label_name}' samples → {DATASET_PATH}")


if __name__ == "__main__":
    args = parse_args()
    collect(args.label, args.samples, args.countdown)
