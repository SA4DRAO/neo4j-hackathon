"""Offline tests for the LangGraph memory routing and persistence nodes."""
import unittest
from unittest.mock import Mock, patch

import agent
import classifier


class AgentGraphTests(unittest.TestCase):
    def test_retrieve_filters_open_tickets_by_classified_product(self):
        state = {
            "customer_id": "customer-1",
            "classification": {
                "intent": "FOLLOW_UP",
                "target_product": "Camera",
                "ticket_action": "UPDATE",
                "is_resolved": False,
            },
        }
        ticket = {"ticket_id": "SUP-1", "product": "Camera"}
        record = Mock()
        record.data.return_value = ticket

        with patch("agent.query", return_value=[record]) as query:
            result = agent.retrieve(state)

        self.assertEqual(result, {"history": [ticket]})
        self.assertEqual(query.call_args.kwargs["id"], "customer-1")
        self.assertEqual(query.call_args.kwargs["product"], "Camera")
        self.assertIn("$product IS NULL OR p.name = $product", query.call_args.args[0])

    def test_new_issue_creates_and_updates_a_ticket(self):
        state = {
            "customer_id": "customer-1",
            "message": "The camera will not turn on.",
            "classification": {
                "intent": "NEW_ISSUE",
                "target_product": "Camera",
                "ticket_action": "CREATE",
                "is_resolved": False,
            },
            "history": [],
        }

        with patch("agent.query", return_value=[]) as query:
            result = agent.mutate(state)

        self.assertRegex(result["ticket_id"], r"^SUP-[0-9a-f]{12}$")
        self.assertEqual(query.call_count, 2)
        self.assertIn("CREATE (t:Ticket", query.call_args_list[0].args[0])
        self.assertIn("CREATE (i:Interaction", query.call_args_list[1].args[0])
        self.assertEqual(query.call_args_list[1].kwargs["ticket_id"], result["ticket_id"])

    def test_resolved_follow_up_reuses_and_closes_retrieved_ticket(self):
        state = {
            "customer_id": "customer-1",
            "message": "That fixed it, thanks.",
            "classification": {
                "intent": "FOLLOW_UP",
                "target_product": "Camera",
                "ticket_action": "CLOSE",
                "is_resolved": True,
            },
            "history": [{"ticket_id": "TICKET-42"}],
        }

        with patch("agent.query", return_value=[]) as query:
            result = agent.mutate(state)

        self.assertEqual(result, {"ticket_id": "TICKET-42"})
        query.assert_called_once()
        self.assertTrue(query.call_args.kwargs["resolved"])
        self.assertEqual(query.call_args.kwargs["ticket_id"], "TICKET-42")

    def test_load_session_returns_only_the_current_customers_pending_slots(self):
        state = {"customer_id": "customer-1", "session_key": "customer-1:web"}
        record = Mock()
        record.data.return_value = {"pending_intent": "NEW_ISSUE"}

        with patch("agent.query", return_value=[record]) as query:
            result = agent.load_session(state)

        self.assertEqual(result, {"session": {"pending_intent": "NEW_ISSUE"}})
        self.assertEqual(query.call_args.kwargs, {
            "customer_id": "customer-1",
            "session_key": "customer-1:web",
            "intent_key": "pending_intent",
            "product_key": "pending_product",
            "issue_key": "pending_issue_type",
        })

    def test_clarify_persists_only_high_confidence_slots_in_the_customer_session(self):
        state = {
            "customer_id": "customer-1",
            "session_id": "web",
            "session_key": "customer-1:web",
            "session": {},
            "classification": {
                "intent": "NEW_ISSUE",
                "intent_confidence": 0.99,
                "target_product": "Camera",
                "product_confidence": 0.30,
                "issue_type": "ACCESS_DENIED",
                "issue_confidence": 0.95,
                "clarification_question": "Which specific product are you having trouble with?",
            },
        }

        with patch("agent.query") as query:
            result = agent.clarify(state)

        self.assertEqual(result, {"response": state["classification"]["clarification_question"]})
        self.assertEqual(query.call_args.kwargs["pending_intent"], "NEW_ISSUE")
        self.assertIsNone(query.call_args.kwargs["pending_product"])
        self.assertEqual(query.call_args.kwargs["pending_issue_type"], "ACCESS_DENIED")
        self.assertIn("HAS_ACTIVE_SESSION", query.call_args.args[0])

    def test_mutate_clears_session_after_ticket_update(self):
        state = {
            "customer_id": "customer-1",
            "session_key": "customer-1:web",
            "message": "That fixed it, thanks.",
            "classification": {
                "intent": "FOLLOW_UP",
                "target_product": "Camera",
                "ticket_action": "CLOSE",
                "is_resolved": True,
            },
            "history": [{"ticket_id": "TICKET-42"}],
        }

        with patch("agent.query", return_value=[]) as query:
            agent.mutate(state)

        self.assertEqual(query.call_count, 2)
        self.assertIn("DETACH DELETE s", query.call_args_list[1].args[0])
        self.assertEqual(query.call_args_list[1].kwargs["session_key"], "customer-1:web")

    def test_low_confidence_new_issue_is_gated_with_a_clarification(self):
        decision = {
            "intent": "NEW_ISSUE", "intent_confidence": 0.99,
            "target_product": "Camera", "product_confidence": 0.40,
            "issue_type": "ACCESS_DENIED", "issue_confidence": 0.30,
            "ticket_action": "CREATE", "action_confidence": 0.99,
            "response_mode": "TROUBLESHOOT", "response_confidence": 0.99,
            "next_step": "ASK_FOR_DETAILS", "next_step_confidence": 0.99,
            "is_resolved": False,
        }

        with patch("classifier._jev_decide", return_value=decision):
            result = classifier.classify("It is not working.", [], ["Camera"])

        self.assertTrue(result.needs_clarification)
        self.assertEqual(
            result.clarification_question,
            "Which product are you using, and is this an access, billing, or technical issue?",
        )

    def test_high_confidence_new_issue_can_create_memory(self):
        decision = {
            "intent": "NEW_ISSUE", "intent_confidence": 0.99,
            "target_product": "Camera", "product_confidence": 0.99,
            "issue_type": "ACCESS_DENIED", "issue_confidence": 0.99,
            "ticket_action": "CREATE", "action_confidence": 0.99,
            "response_mode": "TROUBLESHOOT", "response_confidence": 0.99,
            "next_step": "CHECK_ACCOUNT_ACCESS", "next_step_confidence": 0.99,
            "is_resolved": False,
        }

        with patch("classifier._jev_decide", return_value=decision):
            result = classifier.classify("I cannot sign in to Camera.", [], ["Camera"])

        self.assertFalse(result.needs_clarification)
        self.assertEqual(result.target_product, "Camera")
        self.assertEqual(result.issue_type, "ACCESS_DENIED")

    def test_verified_session_slots_complete_the_next_message(self):
        decision = {
            "intent": "NEW_ISSUE", "intent_confidence": 0.20,
            "target_product": "Camera", "product_confidence": 0.99,
            "issue_type": "BILLING", "issue_confidence": 0.99,
            "ticket_action": "CREATE", "action_confidence": 0.99,
            "response_mode": "TROUBLESHOOT", "response_confidence": 0.99,
            "next_step": "CHECK_BILLING", "next_step_confidence": 0.99,
            "is_resolved": False,
        }

        with patch("classifier._jev_decide", return_value=decision):
            result = classifier.classify(
                "It is Camera.",
                [],
                ["Camera"],
                pending={"pending_intent": "NEW_ISSUE"},
            )

        self.assertFalse(result.needs_clarification)
        self.assertEqual(result.intent, "NEW_ISSUE")
        self.assertEqual(result.target_product, "Camera")
        self.assertEqual(result.issue_type, "BILLING")

    def test_semantic_memory_confirms_a_selected_product(self):
        decision = {
            "intent": "FOLLOW_UP", "intent_confidence": 0.98,
            "target_product": "Camera", "product_confidence": 0.30,
            "issue_type": "TECHNICAL_BUG", "issue_confidence": 0.99,
            "ticket_action": "UPDATE", "action_confidence": 0.98,
            "response_mode": "TROUBLESHOOT", "response_confidence": 0.99,
            "next_step": "CHECK_CONNECTION", "next_step_confidence": 0.99,
            "is_resolved": False,
        }

        with patch("classifier._jev_decide", return_value=decision):
            result = classifier.classify(
                "It is still failing.",
                [],
                ["Camera"],
                semantic_memory=[{"content": "Camera | TECHNICAL_BUG", "score": 0.60}],
            )

        self.assertFalse(result.needs_clarification)
        self.assertGreaterEqual(result.product_confidence, classifier.CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
