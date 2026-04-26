"""
RiskPulse AI - LLM Explanation Layer
Generates human-readable fraud explanations using a template-first approach,
with optional LLM augmentation.  Decision is NEVER delegated to the LLM.
"""

import os
import textwrap
from typing import Any, Dict, List


# ── Template-based fallback (always works, no API key needed) ─────────────────
def _template_explanation(
    decision: str,
    fraud_score: float,
    triggered_rules: List[str],
    risk_factors: Dict[str, Any],
) -> str:
    pct = int(fraud_score * 100)
    amount = risk_factors.get("amount", 0)
    device = risk_factors.get("device_type", "unknown")
    country = risk_factors.get("country_code", "N/A")

    if decision == "BLOCK":
        intro = (
            f"⛔ This transaction has been BLOCKED with a fraud risk score of {pct}%. "
            "Multiple high-severity signals were detected simultaneously."
        )
    elif decision == "REVIEW":
        intro = (
            f"⚠️ This transaction requires MANUAL REVIEW (fraud risk: {pct}%). "
            "One or more suspicious patterns were identified."
        )
    else:
        intro = (
            f"✅ This transaction appears LEGITIMATE (fraud risk: {pct}%). "
            "No significant risk signals were detected."
        )

    details = ""
    if triggered_rules:
        rule_list = "\n".join(f"  • {r}" for r in triggered_rules)
        details = f"\n\nRisk signals detected:\n{rule_list}"

    context = (
        f"\n\nTransaction context: ${amount:,.2f} via {device} device from {country}."
    )

    recommendation = {
        "BLOCK": (
            "\n\nRecommendation: Immediately decline the transaction and alert the "
            "cardholder. Consider temporarily freezing the card pending verification."
        ),
        "REVIEW": (
            "\n\nRecommendation: Hold the transaction and attempt cardholder "
            "verification via OTP or a callback before proceeding."
        ),
        "APPROVE": (
            "\n\nRecommendation: Proceed with the transaction. Continue standard "
            "monitoring for subsequent activity."
        ),
    }.get(decision, "")

    return textwrap.dedent(intro + details + context + recommendation).strip()


# ── LLM-powered explanation (optional — requires ANTHROPIC_API_KEY) ───────────
def _llm_explanation(
    decision: str,
    fraud_score: float,
    anomaly_score: float,
    triggered_rules: List[str],
    risk_factors: Dict[str, Any],
) -> str | None:
    """
    Calls Claude to generate a concise, plain-English fraud explanation.
    Returns None if the API key is absent or the call fails.
    The LLM is strictly an EXPLAINER — it never overrides the ML decision.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=api_key)

        rules_text = (
            "\n".join(f"- {r}" for r in triggered_rules)
            if triggered_rules
            else "None"
        )

        prompt = f"""You are a fraud analyst assistant for a financial institution.
A transaction has been analysed by our ML engine and rule system.
Your job is to write a clear, concise, 2–4 sentence explanation for a bank investigator.

Decision: {decision}
Fraud score: {fraud_score:.2%}
Anomaly score: {anomaly_score:.2%}
Triggered risk rules:
{rules_text}

Transaction details:
- Amount: ${risk_factors.get('amount', 0):,.2f}
- Device: {risk_factors.get('device_type', 'unknown')}
- Country: {risk_factors.get('country_code', 'N/A')}
- Card age: {risk_factors.get('card_age_days', '?')} days
- Transactions in last hour: {risk_factors.get('tx_count_1h', 0)}
- Transaction hour: {risk_factors.get('hour', '?')}:00

Write 2–4 sentences explaining WHY the decision is {decision}.
Be specific, factual, and professional. Do NOT repeat the numbers verbatim — synthesise the story.
Do NOT suggest a different decision than {decision}."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    except Exception:
        return None


# ── Public function ───────────────────────────────────────────────────────────
def generate_explanation(
    decision: str,
    fraud_score: float,
    anomaly_score: float,
    triggered_rules: List[str],
    risk_factors: Dict[str, Any],
) -> str:
    """
    Returns the best available explanation:
    1. LLM (if ANTHROPIC_API_KEY is set and call succeeds)
    2. Template fallback (always available)
    """
    llm_result = _llm_explanation(
        decision, fraud_score, anomaly_score, triggered_rules, risk_factors
    )
    if llm_result:
        return llm_result

    return _template_explanation(decision, fraud_score, triggered_rules, risk_factors)
