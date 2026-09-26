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
    ticket_action: Literal["CREATE", "UPDATE", "CLOSE", "NONE"]
    response_mode: Literal["CLARIFY", "TROUBLESHOOT", "STATUS", "ESCALATE"]
    next_step: Literal[
        "ASK_FOR_DETAILS",
        "RETRY_ACTIVATION",
        "CHECK_CONNECTION",
        "CHECK_ACCOUNT_ACCESS",
        "CHECK_BILLING",
        "HUMAN_HANDOFF",
        "NONE",
    ]
    intent_confidence: float = Field(ge=0, le=1)
    product_confidence: float = Field(ge=0, le=1)
    issue_confidence: float = Field(ge=0, le=1)
    action_confidence: float = Field(ge=0, le=1)
    response_confidence: float = Field(ge=0, le=1)
    next_step_confidence: float = Field(ge=0, le=1)
    is_resolved: bool
    needs_clarification: bool
    clarification_question: str | None


def _jev_decide(message, products, semantic_memory=None):
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
        "ticket_action": {
            "type": "choice",
            "instructions": "What graph mutation should the support agent perform?",
            "criteria": {
                "CREATE": "Create a ticket for a clearly new issue.",
                "UPDATE": "Add the message to a known or ongoing issue without closing it.",
                "CLOSE": "Close an existing ticket only when the customer explicitly says it is fixed.",
                "NONE": "Do not create or update a ticket for general questions or missing details.",
            },
        },
        "response_mode": {
            "type": "choice",
            "instructions": "What bounded response strategy should the prose model follow?",
            "criteria": {
                "CLARIFY": "Ask one focused question because required support details are missing.",
                "TROUBLESHOOT": "Give the selected practical support step for a product issue.",
                "STATUS": "Answer a graph-backed status or general account question without troubleshooting.",
                "ESCALATE": "Acknowledge urgency or churn risk and direct the customer to human support.",
            },
        },
        "next_step": {
            "type": "choice",
            "instructions": "Choose exactly one approved next step, or NONE when no step is needed.",
            "criteria": {
                "ASK_FOR_DETAILS": "Ask for the missing product, issue category, or ticket context.",
                "RETRY_ACTIVATION": "Ask the customer to retry a product activation.",
                "CHECK_CONNECTION": "Ask the customer to verify their internet connection.",
                "CHECK_ACCOUNT_ACCESS": "Ask the customer to verify sign-in or account access.",
                "CHECK_BILLING": "Ask the customer to review payment or billing information.",
                "HUMAN_HANDOFF": "Direct the customer to human support or escalation.",
                "NONE": "No action is required beyond a direct status or informational answer.",
            },
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
            "state": {
                "message": message,
                "purchased_products": products,
                "semantic_memory": semantic_memory or [],
            },
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
        "ticket_action": answers["ticket_action"]["choice"],
        "action_confidence": answers["ticket_action"]["confidence"],
        "response_mode": answers["response_mode"]["choice"],
        "response_confidence": answers["response_mode"]["confidence"],
        "next_step": answers["next_step"]["choice"],
        "next_step_confidence": answers["next_step"]["confidence"],
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


def classify(message, candidates, purchased_products, pending=None, semantic_memory=None):
    """Classify against controlled labels and gate unresolved new support requests."""
    del candidates  # Product choices are limited to verified graph purchases.
    products = sorted(set(purchased_products))
    decision = _jev_decide(message, products, semantic_memory)

    intent = decision["intent"]
    intent_confidence = decision["intent_confidence"]
    target_product = decision["target_product"]
    product_confidence = decision["product_confidence"]
    issue_type = decision["issue_type"]
    issue_confidence = decision["issue_confidence"]
    ticket_action = decision["ticket_action"]
    action_confidence = decision["action_confidence"]
    response_mode = decision["response_mode"]
    response_confidence = decision["response_confidence"]
    next_step = decision["next_step"]
    next_step_confidence = decision["next_step_confidence"]

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

    semantic_memory = semantic_memory or []
    if product_confidence < CONFIDENCE_THRESHOLD and target_product:
        corroborated = any(
            item.get("score", 0) >= 0.5
            and target_product.casefold() in str(item.get("content", "")).casefold()
            for item in semantic_memory
        )
        if corroborated:
            product_confidence = CONFIDENCE_THRESHOLD

    needs_clarification = response_mode == "CLARIFY" or intent_confidence < CONFIDENCE_THRESHOLD or action_confidence < CONFIDENCE_THRESHOLD or (
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
        ticket_action=ticket_action,
        response_mode=response_mode,
        next_step=next_step,
        intent_confidence=intent_confidence,
        product_confidence=product_confidence,
        issue_confidence=issue_confidence,
        action_confidence=action_confidence,
        response_confidence=response_confidence,
        next_step_confidence=next_step_confidence,
        is_resolved=decision["is_resolved"],
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
    )
