# =============================================================================
# src/model/predictor.py — Runtime inference wrapper
#
# Loads the saved pipeline (scaler + RF) once and exposes a simple
# predict() call that the live-detection loop can call every frame.
# =============================================================================

from __future__ import annotations

import os
import sys
import numpy as np
import joblib
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import MODEL_PATH, LABEL_NAMES


class DrowsinessPredictor:
    """
    Loads the trained pipeline and provides per-frame predictions.

    Usage
    -----
        predictor = DrowsinessPredictor()
        predictor.load()
        label, confidence = predictor.predict(feature_vector)
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self._model_path = model_path
        self._pipeline   = None

    def load(self) -> None:
        """Load the pipeline from disk. Call once before predict()."""
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"No trained model at: {self._model_path}\n"
                "Run training first:  python -m src.model.trainer"
            )
        self._pipeline = joblib.load(self._model_path)
        print(f"[INFO] Model loaded from {self._model_path}")

    def predict(self, feature_array: np.ndarray) -> Tuple[int, float]:
        """
        Predict label and confidence for one feature vector.

        Parameters
        ----------
        feature_array : (1, 7) numpy array from FeatureVector.to_model_input()

        Returns
        -------
        (label_int, confidence)
          label_int  : 0 = AWAKE, 1 = DROWSY
          confidence : probability of the predicted class [0, 1]
        """
        if self._pipeline is None:
            raise RuntimeError("Call DrowsinessPredictor.load() first.")

        label_int  = int(self._pipeline.predict(feature_array)[0])
        proba      = self._pipeline.predict_proba(feature_array)[0]
        confidence = float(proba[label_int])
        return label_int, confidence

    def predict_proba_drowsy(self, feature_array: np.ndarray) -> float:
        """
        Return only the probability of the DROWSY class.
        Used by the fatigue-score rolling window.
        """
        if self._pipeline is None:
            raise RuntimeError("Call DrowsinessPredictor.load() first.")
        return float(self._pipeline.predict_proba(feature_array)[0][1])

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
