"""
RiskPulse AI — FastAPI Backend
================================
Run:  uvicorn main:app --reload --port 8000
"""

import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from features import extract_features, feature_summary
from model import predict_anomaly_score
from decision import make_decision
from llm_explainer import generate_explanation

# ── In-memory fraud log (last 200 transactions) ───────────────────────────────
FRAUD_LOG: List[Dict] = []
MAX_LOG = 200

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RiskPulse AI",
    description="Real-time fraud detection powered by ML + rule engine + LLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    user_id: str = Field(default="user_anonymous", description="Stable user ID")
    device_type: str = Field(default="unknown", description="mobile | desktop | tablet | unknown")
    country_code: str = Field(default="US", max_length=2, description="ISO-3166-1 alpha-2")
    latitude: float = Field(default=40.7128, ge=-90, le=90)
    longitude: float = Field(default=-74.006, ge=-180, le=180)
    tx_count_1h: int = Field(default=0, ge=0, description="Transactions in last hour")
    card_age_days: int = Field(default=365, ge=0, description="Days since card was issued")
    hour: Optional[int] = Field(default=None, ge=0, le=23, description="Hour of day (0-23)")
    merchant_name: Optional[str] = Field(default=None, description="Optional merchant name")
    merchant_category: Optional[str] = Field(default=None, description="Optional merchant category")

    @validator("device_type")
    def normalise_device(cls, v):
        return v.strip().lower()

    @validator("country_code")
    def upper_country(cls, v):
        return v.strip().upper()

    class Config:
        schema_extra = {
            "example": {
                "amount": 4500.00,
                "user_id": "user_9182",
                "device_type": "mobile",
                "country_code": "NG",
                "latitude": 6.5244,
                "longitude": 3.3792,
                "tx_count_1h": 6,
                "card_age_days": 12,
                "hour": 3,
                "merchant_name": "Global Electronics",
                "merchant_category": "electronics",
            }
        }


class FraudResponse(BaseModel):
    transaction_id: str
    timestamp: str
    fraud_score: float
    anomaly_score: float
    decision: str
    explanation: str
    triggered_rules: List[str]
    risk_factors: Dict[str, Any]
    processing_ms: float


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "RiskPulse AI", "version": "1.0.0"}


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/detect_fraud", response_model=FraudResponse, tags=["fraud"])
def detect_fraud(tx: TransactionRequest):
    t0 = time.perf_counter()

    payload = tx.dict()

    # 1. Feature engineering
    features = extract_features(payload)
    summary = feature_summary(payload)

    # 2. ML anomaly score
    anomaly_score = predict_anomaly_score(features)

    # 3. Rule engine + final decision
    result = make_decision(anomaly_score, summary)

    # 4. LLM explanation
    explanation = generate_explanation(
        result.decision,
        result.fraud_score,
        result.anomaly_score,
        result.triggered_rules,
        result.risk_factors,
    )

    processing_ms = round((time.perf_counter() - t0) * 1000, 2)
    tx_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    response = FraudResponse(
        transaction_id=tx_id,
        timestamp=ts,
        fraud_score=result.fraud_score,
        anomaly_score=result.anomaly_score,
        decision=result.decision,
        explanation=explanation,
        triggered_rules=result.triggered_rules,
        risk_factors=result.risk_factors,
        processing_ms=processing_ms,
    )

    # Log to in-memory store
    log_entry = response.dict()
    log_entry["input"] = {
        "amount": tx.amount,
        "device_type": tx.device_type,
        "country_code": tx.country_code,
        "merchant_name": tx.merchant_name,
    }
    FRAUD_LOG.insert(0, log_entry)
    if len(FRAUD_LOG) > MAX_LOG:
        FRAUD_LOG.pop()

    return response


@app.get("/logs", tags=["fraud"])
def get_logs(limit: int = 20):
    """Return the most recent fraud detection results."""
    return {"count": len(FRAUD_LOG), "logs": FRAUD_LOG[:limit]}


@app.get("/stats", tags=["fraud"])
def get_stats():
    """Aggregate statistics over the in-memory log."""
    if not FRAUD_LOG:
        return {"total": 0, "approved": 0, "review": 0, "blocked": 0}

    decisions = [e["decision"] for e in FRAUD_LOG]
    scores = [e["fraud_score"] for e in FRAUD_LOG]

    return {
        "total": len(FRAUD_LOG),
        "approved": decisions.count("APPROVE"),
        "review": decisions.count("REVIEW"),
        "blocked": decisions.count("BLOCK"),
        "avg_fraud_score": round(sum(scores) / len(scores), 4),
        "max_fraud_score": round(max(scores), 4),
    }


@app.delete("/logs", tags=["fraud"])
def clear_logs():
    FRAUD_LOG.clear()
    return {"cleared": True}
