"""Jev classification via OpenRouter's Decisions API, with a deterministic confidence gate."""
import os
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from clients import JEV_MODEL, OPENROUTER_API_KEY


CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.85"))
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

_ISSUE_CRITERIA = {
    "ACCESS_DENIED": "Login, permissions, or account access problem.",
    "BILLING": "Charges, invoices, or payment problem.",
    "TECHNICAL_BUG": "The product is broken or behaving unexpectedly.",
    "NONE": "Not a new issue or escalation, or no issue category applies.",
}


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["NEW_ISSUE", "FOLLOW_UP", "CHITCHAT", "ESCALATION"]
    target_product: str | None
    issue_type: Literal["ACCESS_DENIED", "BILLING", "TECHNICAL_BUG"] | None
    intent_confidence: float = Field(ge=0, le=1)
    product_confidence: float = Field(ge=0, le=1)
    issue_confidence: float = Field(ge=0, le=1)
    is_resolved: bool
    needs_clarification: bool
    clarification_question: str | None


def _jev_decide(message, products):
    """Ask Jev's Decisions API for a raw, per-field decision. No pending-slot merge or gating here."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env to call Jev.")

    product_criteria = {product: f"The message concerns {product}." for product in products}
    product_criteria["NONE"] = "No purchased product is referenced."
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "What is the customer's core support intent?",
            "criteria": {
                "NEW_ISSUE": "A newly reported problem.",
                "FOLLOW_UP": "Refers to an existing, already-reported issue.",
                "ESCALATION": "Explicit strong frustration or intent to leave or cancel.",
                "CHITCHAT": "Anything else, including general questions.",
            },
        },
        "target_product": {
            "type": "choice",
            "instructions": "Which purchased product does the message concern, if any?",
            "criteria": product_criteria,
        },
        "issue_type": {
            "type": "choice",
            "instructions": "What category of issue is this, if any?",
            "criteria": _ISSUE_CRITERIA,
        },
        "is_resolved": {
            "type": "noul",
            "instructions": "Does the customer say their issue is now fixed or resolved?",
            "criteria": {
                "true": "The customer explicitly says the issue is fixed or resolved.",
                "false": "The customer has not said the issue is fixed.",
            },
        },
    }
    response = requests.post(
        DECISIONS_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": JEV_MODEL,
            "state": {"message": message, "purchased_products": products},
            "questions": questions,
        },
        timeout=30,
    )
    response.raise_for_status()
    answers = response.json()["answers"]
    target_product = answers["target_product"]["choice"]
    issue_type = answers["issue_type"]["choice"]
    return {
        "intent": answers["intent"]["choice"],
        "intent_confidence": answers["intent"]["confidence"],
        "target_product": None if target_product == "NONE" else target_product,
        "product_confidence": answers["target_product"]["confidence"],
        "issue_type": None if issue_type == "NONE" else issue_type,
        "issue_confidence": answers["issue_type"]["confidence"],
        "is_resolved": answers["is_resolved"]["noul"] >= 0.5,
    }


def _clarification(intent_confidence, product_confidence, issue_confidence):
    if intent_confidence < CONFIDENCE_THRESHOLD:
        return "Is this a new issue, a follow-up to an existing ticket, or an escalation?"
    missing_product = product_confidence < CONFIDENCE_THRESHOLD
    missing_issue = issue_confidence < CONFIDENCE_THRESHOLD
    if missing_product and missing_issue:
        return (
            "Which product are you using, and is this an access, billing, or technical issue?"
        )
    if missing_product:
        return "Which specific product are you having trouble with?"
    return "Is this an access issue, a billing issue, or a technical problem?"


def classify(message, candidates, purchased_products, pending=None):
    """Classify against controlled labels and gate unresolved new support requests."""
    del candidates  # Product choices are limited to verified graph purchases.
    products = sorted(set(purchased_products))
    decision = _jev_decide(message, products)

    intent = decision["intent"]
    intent_confidence = decision["intent_confidence"]
    target_product = decision["target_product"]
    product_confidence = decision["product_confidence"]
    issue_type = decision["issue_type"]
    issue_confidence = decision["issue_confidence"]

    pending = pending or {}
    if intent_confidence < CONFIDENCE_THRESHOLD and pending.get("pending_intent"):
        intent = pending["pending_intent"]
        intent_confidence = CONFIDENCE_THRESHOLD
    if product_confidence < CONFIDENCE_THRESHOLD and pending.get("pending_product"):
        target_product = pending["pending_product"]
        product_confidence = CONFIDENCE_THRESHOLD
    if issue_confidence < CONFIDENCE_THRESHOLD and pending.get("pending_issue_type"):
        issue_type = pending["pending_issue_type"]
        issue_confidence = CONFIDENCE_THRESHOLD

    needs_clarification = intent_confidence < CONFIDENCE_THRESHOLD or (
        intent in {"NEW_ISSUE", "ESCALATION"}
        and (
            product_confidence < CONFIDENCE_THRESHOLD
            or issue_confidence < CONFIDENCE_THRESHOLD
        )
    )
    clarification_question = (
        _clarification(intent_confidence, product_confidence, issue_confidence)
        if needs_clarification
        else None
    )
    return Classification(
        intent=intent,
        target_product=target_product,
        issue_type=issue_type,
        intent_confidence=intent_confidence,
        product_confidence=product_confidence,
        issue_confidence=issue_confidence,
        is_resolved=decision["is_resolved"],
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
    )
