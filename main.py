# =============================================================================
# main.py — Entry point for the Drowsiness Detection System
#
# Usage:
#   python main.py              → live detection (requires trained model)
#   python main.py --check      → pre-flight environment check only
# =============================================================================

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_PATH, DATASET_PATH



def parse_args():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection")
    parser.add_argument("--check", action="store_true",
                        help="Run environment check and exit.")
    return parser.parse_args()


def preflight_check() -> bool:
    """
    Verify all dependencies and project artefacts are in place.
    Returns True if everything is ready to run.
    """
    print("\n── Pre-flight Check ─────────────────────────────────")
    all_ok = True

    # 1. Python packages
    packages = {
        "cv2":       "opencv-python",
        "mediapipe": "mediapipe",
        "numpy":     "numpy",
        "pandas":    "pandas",
        "sklearn":   "scikit-learn",
        "joblib":    "joblib",
        "scipy":     "scipy",
    }
    for module, pkg in packages.items():
        try:
            __import__(module)
            print(f"  [OK]  {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg} — run: pip install {pkg}")
            all_ok = False

    # 2. Dataset
    if os.path.exists(DATASET_PATH):
        import pandas as pd
        df = pd.read_csv(DATASET_PATH)
        print(f"  [OK]  Dataset found ({len(df)} rows) → {DATASET_PATH}")
    else:
        print(f"  [WARN] No dataset yet — run data_collector.py first")

    # 3. Trained model
    if os.path.exists(MODEL_PATH):
        print(f"  [OK]  Model found → {MODEL_PATH}")
    else:
        print(f"  [WARN] No trained model yet — run: python -m src.model.trainer")

    print("─────────────────────────────────────────────────────\n")
    return all_ok


def run_live_detection():
    """
    Full live-detection pipeline.
    """
    import cv2
    import time

    from src.vision.camera import Camera
    from src.vision.landmark_detector import LandmarkDetector
    from src.features.extractor import FeatureExtractor
    from src.model.predictor import DrowsinessPredictor

    from src.dashboard.hud import HUD
    from src.fatigue.fatigue_score import FatigueScorer

    from src.analytics.blink_counter import BlinkCounter
    from src.analytics.yawn_counter import YawnCounter
    from src.analytics.session_stats import SessionStats
    from src.alerts.alert_manager import AlertManager

    predictor = DrowsinessPredictor()
    predictor.load()

    extractor = FeatureExtractor()

    hud = HUD()
    fatigue = FatigueScorer(window_size=30)

    blink_counter = BlinkCounter()
    yawn_counter = YawnCounter()

    stats = SessionStats()

    alert_manager = AlertManager()

    prev_time = time.time()

    print("[INFO] Starting live detection. Press 'q' to quit.")

    with Camera() as cam, LandmarkDetector() as detector:

        while True:

            ret, frame = cam.read()

            if not ret:
                continue

            result = detector.process(frame)
            annotated = detector.draw_landmarks(frame, result)

            current_time = time.time()
            fps = 1 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            if result.face_detected:

                fv = extractor.extract(result)

                label_int, confidence = predictor.predict(
                    fv.to_model_input()
                )

                probability = predictor.predict_proba_drowsy(
                    fv.to_model_input()
                )

                fatigue_score, alert_level = fatigue.update(probability)

                blinks = blink_counter.update(fv.ear_avg)

                yawns = yawn_counter.update(fv.mar)

                stats.set_blinks(blinks)
                stats.set_yawns(yawns)

                stats.update_fatigue(fatigue_score)
                if alert_manager.update(fatigue_score):
                    stats.increment_alerts()

                prediction = "DROWSY" if label_int else "AWAKE"

                annotated = hud.draw(
                    frame=annotated,
                    ear=fv.ear_avg,
                    mar=fv.mar,
                    pitch=fv.pitch,
                    prediction=prediction,
                    confidence=confidence * 100,
                    fatigue_score=fatigue_score,
                    alert_level=alert_level,
                    fps=fps,
                    blinks=blinks,
                    yawns=yawns,
                )

            else:

                cv2.putText(
                    annotated,
                    "No Face Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow(
                "Driver Monitoring System",
                annotated,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


def main():
    print("=" * 52)
    print("  Driver Drowsiness Detection System")
    print("=" * 52)

    args = parse_args()

    if args.check:
        preflight_check()
        return

    ok = preflight_check()
    if not ok:
        print("[ERROR] Fix missing dependencies before running.")
        sys.exit(1)

    run_live_detection()


if __name__ == "__main__":
    main()
