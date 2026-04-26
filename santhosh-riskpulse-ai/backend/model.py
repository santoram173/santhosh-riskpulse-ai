"""
RiskPulse AI - ML Model Layer
Isolation Forest (anomaly) + Rule-based scoring + Optional XGBoost
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
IFOREST_PATH = os.path.join(MODEL_DIR, "iforest.pkl")


# ── Singleton loader ─────────────────────────────────────────────────────────
class ModelRegistry:
    _scaler: StandardScaler = None
    _iforest: IsolationForest = None

    @classmethod
    def get_models(cls) -> Tuple[StandardScaler, IsolationForest]:
        if cls._scaler is None or cls._iforest is None:
            cls._scaler, cls._iforest = _load_or_train()
        return cls._scaler, cls._iforest


# ── Training on synthetic data ────────────────────────────────────────────────
def _generate_synthetic_data(n_samples: int = 2000) -> np.ndarray:
    """
    Synthetic feature matrix for training:
    [amount_log, hour_sin, hour_cos, location_entropy,
     device_risk, velocity_score, country_risk, card_age_days]
    """
    rng = np.random.default_rng(42)

    # Normal transactions
    normal = rng.normal(
        loc=[3.5, 0.0, 1.0, 0.4, 0.2, 0.3, 0.2, 0.6],
        scale=[0.8, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.3],
        size=(int(n_samples * 0.95), 8),
    )

    # Anomalous transactions
    anomalous = rng.normal(
        loc=[7.0, 0.0, 1.0, 0.9, 0.8, 0.9, 0.8, 0.1],
        scale=[1.5, 0.8, 0.8, 0.15, 0.15, 0.15, 0.15, 0.1],
        size=(int(n_samples * 0.05), 8),
    )

    return np.vstack([normal, anomalous])


def _load_or_train() -> Tuple[StandardScaler, IsolationForest]:
    if os.path.exists(SCALER_PATH) and os.path.exists(IFOREST_PATH):
        scaler = joblib.load(SCALER_PATH)
        iforest = joblib.load(IFOREST_PATH)
        return scaler, iforest

    X = _generate_synthetic_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iforest = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iforest.fit(X_scaled)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(iforest, IFOREST_PATH)
    return scaler, iforest


# ── Public API ────────────────────────────────────────────────────────────────
def predict_anomaly_score(feature_vector: np.ndarray) -> float:
    """
    Returns a normalised anomaly score in [0, 1].
    Higher = more anomalous.
    """
    scaler, iforest = ModelRegistry.get_models()
    x = scaler.transform(feature_vector.reshape(1, -1))

    # Raw score: negative → more anomalous
    raw = iforest.score_samples(x)[0]

    # Convert to [0,1]: typical range is roughly [-0.5, 0.1]
    clipped = np.clip(raw, -0.6, 0.1)
    normalised = 1.0 - (clipped + 0.6) / 0.7
    return float(np.clip(normalised, 0.0, 1.0))


# Pre-load models at import time (warm-up)
try:
    ModelRegistry.get_models()
except Exception:
    pass
