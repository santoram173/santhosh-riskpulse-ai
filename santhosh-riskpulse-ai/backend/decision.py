"""
RiskPulse AI - Decision Engine
Combines ML anomaly score + rule-based signals → final fraud_score + decision.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DecisionResult:
    fraud_score: float          # [0, 1]
    anomaly_score: float        # [0, 1] — raw from Isolation Forest
    decision: str               # APPROVE | REVIEW | BLOCK
    triggered_rules: List[str]
    risk_factors: Dict[str, Any]


# ── Threshold config ──────────────────────────────────────────────────────────
BLOCK_THRESHOLD = 0.72
REVIEW_THRESHOLD = 0.42


# ── Individual rules (each returns an additive penalty in [0, 1]) ─────────────
def _rule_large_amount(summary: Dict) -> tuple[float, str | None]:
    amt = summary.get("amount", 0)
    if amt > 10_000:
        return 0.30, f"Very large transaction (${amt:,.2f})"
    if amt > 5_000:
        return 0.15, f"Large transaction (${amt:,.2f})"
    return 0.0, None


def _rule_unusual_hour(summary: Dict) -> tuple[float, str | None]:
    hour = summary.get("hour", 12)
    if hour < 4 or hour >= 23:
        return 0.18, f"Unusual transaction hour ({hour:02d}:00)"
    return 0.0, None


def _rule_high_velocity(summary: Dict) -> tuple[float, str | None]:
    count = summary.get("tx_count_1h", 0)
    if count >= 8:
        return 0.30, f"Very high velocity ({count} txns in last hour)"
    if count >= 4:
        return 0.15, f"Elevated velocity ({count} txns in last hour)"
    return 0.0, None


def _rule_risky_country(summary: Dict) -> tuple[float, str | None]:
    cr = summary.get("country_risk_score", 0.15)
    cc = summary.get("country_code", "US")
    if cr >= 0.80:
        return 0.25, f"High-risk originating country ({cc})"
    if cr >= 0.40:
        return 0.10, f"Elevated-risk country ({cc})"
    return 0.0, None


def _rule_unknown_device(summary: Dict) -> tuple[float, str | None]:
    dr = summary.get("device_risk_score", 0.2)
    dev = summary.get("device_type", "unknown")
    if dr >= 0.65:
        return 0.20, f"Unrecognised or high-risk device ({dev})"
    return 0.0, None


def _rule_new_card(summary: Dict) -> tuple[float, str | None]:
    age = summary.get("card_age_days", 365)
    if age < 7:
        return 0.25, f"Card issued only {age} day(s) ago"
    if age < 30:
        return 0.12, f"Relatively new card ({age} days old)"
    return 0.0, None


def _rule_high_location_entropy(summary: Dict) -> tuple[float, str | None]:
    entropy = summary.get("location_entropy", 0.0)
    if entropy > 0.80:
        return 0.20, "Unusual merchant location for this user"
    if entropy > 0.55:
        return 0.08, "Slightly unusual merchant location"
    return 0.0, None


RULES = [
    _rule_large_amount,
    _rule_unusual_hour,
    _rule_high_velocity,
    _rule_risky_country,
    _rule_unknown_device,
    _rule_new_card,
    _rule_high_location_entropy,
]


# ── Main decision function ────────────────────────────────────────────────────
def make_decision(anomaly_score: float, feature_summary: Dict[str, Any]) -> DecisionResult:
    """
    Combines the ML anomaly score with rule-based penalties to produce
    a final fraud_score and deterministic decision.
    """

    rule_penalty = 0.0
    triggered: List[str] = []

    for rule in RULES:
        penalty, label = rule(feature_summary)
        if penalty > 0 and label:
            rule_penalty += penalty
            triggered.append(label)

    # Weighted blend: 50% ML anomaly, 50% rule engine
    # Cap rule_penalty at 0.8 to avoid overflow
    rule_component = min(rule_penalty, 0.80)
    fraud_score = float(0.50 * anomaly_score + 0.50 * rule_component)
    fraud_score = round(min(fraud_score, 1.0), 4)

    if fraud_score >= BLOCK_THRESHOLD:
        decision = "BLOCK"
    elif fraud_score >= REVIEW_THRESHOLD:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    return DecisionResult(
        fraud_score=fraud_score,
        anomaly_score=round(anomaly_score, 4),
        decision=decision,
        triggered_rules=triggered,
        risk_factors=feature_summary,
    )
