# =============================================================================
# config.py — Central configuration for the Drowsiness Detection System
# All thresholds, paths, and constants live here.
# Changing a value here affects the entire project — no hunting through files.
# =============================================================================

import os

# ── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR   = os.path.join(DATA_DIR, "raw")
PROC_DATA_DIR  = os.path.join(DATA_DIR, "processed")
DATASET_PATH   = os.path.join(PROC_DATA_DIR, "features.csv")

# ── Model Path ────────────────────────────────────────────────────────────────
MODELS_DIR     = os.path.join(BASE_DIR, "models")
MODEL_PATH     = os.path.join(MODELS_DIR, "drowsiness_rf.pkl")
SCALER_PATH    = os.path.join(MODELS_DIR, "scaler.pkl")

# ── Camera Settings ───────────────────────────────────────────────────────────
CAMERA_INDEX   = 0          # 0 = default webcam
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
FPS_TARGET     = 30

# ── MediaPipe Settings ────────────────────────────────────────────────────────
# Confidence thresholds for face detection and tracking
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.7

# ── EAR (Eye Aspect Ratio) Settings ───────────────────────────────────────────
# EAR drops below this threshold when eyes are closing
EAR_THRESHOLD            = 0.25
# Number of consecutive frames with low EAR before flagging drowsiness
EAR_CONSEC_FRAMES        = 20

# ── MAR (Mouth Aspect Ratio) Settings ─────────────────────────────────────────
# MAR rises above this threshold during a yawn
MAR_THRESHOLD            = 0.6

# ── Blink Detection Settings ──────────────────────────────────────────────────
BLINK_MIN_FRAMES = 2

# ── Yawn Detection Settings ───────────────────────────────────────────────────
YAWN_MIN_FRAMES = 8

# ── Head Pose Settings ────────────────────────────────────────────────────────
# Pitch (nodding) angle in degrees — head drooping forward
HEAD_PITCH_THRESHOLD     = 15.0
# Roll (tilting sideways) angle in degrees
HEAD_ROLL_THRESHOLD      = 20.0

# ── Fatigue Score Settings ────────────────────────────────────────────────────
# Rolling window size (in frames) for computing fatigue score
FATIGUE_WINDOW_SIZE      = 60
# Score above this triggers a drowsiness alert
FATIGUE_ALERT_THRESHOLD  = 0.6

# ── Display Settings ──────────────────────────────────────────────────────────
# BGR color tuples for OpenCV drawing
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE  = (255, 255, 255)
FONT         = 0    # cv2.FONT_HERSHEY_SIMPLEX — using int to avoid import here

# ── Data Collection Settings ──────────────────────────────────────────────────
# Labels used during data collection
LABEL_AWAKE  = 0
LABEL_DROWSY = 1
LABEL_NAMES  = {0: "AWAKE", 1: "DROWSY"}

# ── Feature Column Names (used for CSV and model training) ───────────────────
FEATURE_COLUMNS = [
    "ear_left",
    "ear_right",
    "ear_avg",
    "mar",
    "pitch",
    "yaw",
    "roll",
]
TARGET_COLUMN = "label"
