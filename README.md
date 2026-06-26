# Driver Drowsiness Detection System

A real-time computer vision system that detects driver drowsiness using a webcam,
facial landmark analysis, and a machine learning classifier.

## Tech Stack
- **OpenCV** — webcam capture and image rendering
- **MediaPipe** — 468-point face mesh landmark detection
- **NumPy / Pandas** — numerical computation and data handling
- **Scikit-learn** — Random Forest classifier
- **Matplotlib** — evaluation plots

## Features
- Eye Aspect Ratio (EAR) — detects eye closure
- Mouth Aspect Ratio (MAR) — detects yawning
- Head Pose Estimation — detects head drooping
- Rolling Fatigue Score — smoothed drowsiness probability over time
- Audio + visual alert when fatigue threshold is crossed

## Project Structure
```
drowsiness_detection/
├── data/
│   ├── raw/            # Unmodified collected data
│   └── processed/      # features.csv — model-ready dataset
├── src/
│   ├── vision/         # Webcam capture, MediaPipe landmark detection
│   ├── features/       # EAR, MAR, head pose computation
│   ├── model/          # Training and inference
│   └── output/         # Display overlays and alert system
├── models/             # Saved .pkl model and scaler
├── notebooks/          # EDA notebook
├── tests/              # Unit tests
├── config.py           # All constants and thresholds
├── main.py             # Live detection entry point
└── data_collector.py   # Labeled data collection script
```

## Setup
```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
python main.py
```

## Usage
```bash
# Collect training data
python data_collector.py --label awake   --samples 500
python data_collector.py --label drowsy  --samples 500

# Train the model
python src/model/trainer.py

# Run live detection
python main.py
```

## Milestones
- [x] M1 — Project setup and folder structure
- [ ] M2 — Webcam capture with OpenCV
- [ ] M3 — MediaPipe Face Mesh integration
- [ ] M4 — EAR and MAR feature extraction
- [ ] M5 — Head pose estimation
- [ ] M6 — Data collection and CSV generation
- [ ] M7 — Train Random Forest classifier
- [ ] M8 — Model evaluation
- [ ] M9 — Live prediction integration
- [ ] M10 — Fatigue score and alert system
- [ ] M11 — Refactor and finalize
```
