"""
RiskPulse AI - Feature Engineering
Transforms raw transaction payload into ML-ready feature vector.
"""

import math
import hashlib
import numpy as np
from datetime import datetime
from typing import Any, Dict


# ── Device risk mapping ───────────────────────────────────────────────────────
DEVICE_RISK: Dict[str, float] = {
    "mobile": 0.25,
    "desktop": 0.15,
    "tablet": 0.20,
    "unknown": 0.70,
    "other": 0.55,
}

# ── Country risk tier ─────────────────────────────────────────────────────────
HIGH_RISK_COUNTRIES = {
    "NG", "RU", "UA", "CN", "VN", "PK", "BD", "GH", "KE", "SN"
}
MEDIUM_RISK_COUNTRIES = {
    "IN", "BR", "MX", "PH", "ID", "TR", "SA", "EG", "ZA", "AR"
}


def _country_risk(country_code: str) -> float:
    cc = (country_code or "").upper()
    if cc in HIGH_RISK_COUNTRIES:
        return 0.85
    if cc in MEDIUM_RISK_COUNTRIES:
        return 0.45
    return 0.15


def _hour_encoding(hour: int) -> tuple[float, float]:
    """Cyclical sin/cos encoding of hour-of-day."""
    theta = 2 * math.pi * hour / 24
    return math.sin(theta), math.cos(theta)


def _location_entropy(lat: float, lon: float, user_id: str) -> float:
    """
    Synthetic location entropy: deterministic hash-based proxy for
    'how unusual is this lat/lon for this user'.
    Range [0, 1] — higher = more unusual.
    """
    seed = hashlib.md5(f"{user_id}:{round(lat,1)}:{round(lon,1)}".encode()).hexdigest()
    return int(seed[:4], 16) / 65535


def _velocity_score(tx_count_last_hour: int) -> float:
    """Normalised velocity: 0 → low, 1 → very high."""
    return float(np.clip(tx_count_last_hour / 10.0, 0.0, 1.0))


def _amount_log(amount: float) -> float:
    """Log-transform of amount; clamp to [0, 12]."""
    return float(np.clip(math.log1p(max(amount, 0)), 0, 12))


def _card_age_score(card_created_days: int) -> float:
    """New cards are riskier. Returns score in [0,1]; 0 = old card, 1 = brand-new."""
    return float(np.clip(1.0 - card_created_days / 365.0, 0.0, 1.0))


# ── Main extraction function ──────────────────────────────────────────────────
def extract_features(payload: Dict[str, Any]) -> np.ndarray:
    """
    Extracts 8-dimensional feature vector from transaction payload.

    Expected payload keys (all optional — defaults applied):
        amount          float   Transaction amount in USD
        hour            int     Hour of day (0-23); defaults to current hour
        latitude        float   Merchant latitude
        longitude       float   Merchant longitude
        user_id         str     Stable user identifier
        device_type     str     mobile | desktop | tablet | unknown
        country_code    str     ISO-3166-1 alpha-2
        tx_count_1h     int     Transactions in last hour
        card_age_days   int     Days since card was issued
    """

    amount = float(payload.get("amount", 0.0))
    hour = int(payload.get("hour", datetime.utcnow().hour))
    lat = float(payload.get("latitude", 40.7128))
    lon = float(payload.get("longitude", -74.0060))
    user_id = str(payload.get("user_id", "anon"))
    device_type = str(payload.get("device_type", "unknown")).lower()
    country_code = str(payload.get("country_code", "US"))
    tx_count_1h = int(payload.get("tx_count_1h", 0))
    card_age_days = int(payload.get("card_age_days", 365))

    hour_sin, hour_cos = _hour_encoding(hour)

    features = np.array(
        [
            _amount_log(amount),                   # 0: log-amount
            hour_sin,                              # 1: hour sin
            hour_cos,                              # 2: hour cos
            _location_entropy(lat, lon, user_id),  # 3: location entropy
            DEVICE_RISK.get(device_type, 0.55),    # 4: device risk
            _velocity_score(tx_count_1h),          # 5: velocity
            _country_risk(country_code),           # 6: country risk
            _card_age_score(card_age_days),        # 7: card age risk
        ],
        dtype=np.float32,
    )

    return features


def feature_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Human-readable feature breakdown for the explanation layer."""
    amount = float(payload.get("amount", 0.0))
    hour = int(payload.get("hour", datetime.utcnow().hour))
    device_type = str(payload.get("device_type", "unknown")).lower()
    country_code = str(payload.get("country_code", "US")).upper()
    tx_count_1h = int(payload.get("tx_count_1h", 0))
    card_age_days = int(payload.get("card_age_days", 365))
    lat = float(payload.get("latitude", 40.7128))
    lon = float(payload.get("longitude", -74.0060))
    user_id = str(payload.get("user_id", "anon"))

    return {
        "amount": amount,
        "amount_risk": "high" if amount > 5000 else "medium" if amount > 1000 else "low",
        "hour": hour,
        "time_risk": "high" if hour < 5 or hour > 23 else "low",
        "device_type": device_type,
        "device_risk_score": DEVICE_RISK.get(device_type, 0.55),
        "country_code": country_code,
        "country_risk_score": _country_risk(country_code),
        "tx_count_1h": tx_count_1h,
        "velocity_risk": "high" if tx_count_1h >= 5 else "medium" if tx_count_1h >= 2 else "low",
        "card_age_days": card_age_days,
        "card_age_risk": "high" if card_age_days < 30 else "medium" if card_age_days < 90 else "low",
        "location_entropy": _location_entropy(lat, lon, user_id),
    }
