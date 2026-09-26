"""Demo web UI for the graph-memory support agent."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agent import agent
from clients import query


PAGE = Path(__file__).with_name("web.html").read_text(encoding="utf-8")


def records(cypher, **params):
    return [record.data() for record in query(cypher, **params)]


def overview(customer_id, session_id):
    counts = records(
        "MATCH (c:Customer) WITH count(c) AS customers "
        "MATCH (t:Ticket) WITH customers, count(t) AS tickets "
        "MATCH (i:Interaction) WITH customers, tickets, count(i) AS interactions "
        "OPTIONAL MATCH (conversation:Conversation) WITH customers, tickets, interactions, count(conversation) AS conversations "
        "OPTIONAL MATCH (trace:ReasoningTrace) RETURN customers, tickets, interactions, conversations, count(trace) AS traces"
    )[0]
    statuses = records("MATCH (t:Ticket) RETURN t.status AS name, count(*) AS value ORDER BY value DESC")
    customer = records(
        "MATCH (c:Customer {id: $customer_id}) "
        "OPTIONAL MATCH (c)-[:PURCHASED]->(p:Product) "
        "RETURN c.id AS id, c.name AS name, collect(DISTINCT p.name) AS products",
        customer_id=customer_id,
    )
    active_tickets = records(
        "MATCH (:Customer {id: $customer_id})-[:OPENED]->(t:Ticket) "
        "OPTIONAL MATCH (t)-[:REGARDS]->(p:Product) "
        "RETURN t.id AS id, t.subject AS subject, t.status AS status, t.type AS type, p.name AS product "
        "ORDER BY coalesce(t.created_at, datetime({epochMillis: 0})) DESC LIMIT 8",
        customer_id=customer_id,
    )
    interactions = records(
        "MATCH (:Customer {id: $customer_id})-[:HAD_INTERACTION]->(i:Interaction) "
        "OPTIONAL MATCH (i)-[:UPDATES]->(t:Ticket) "
        "RETURN i.summary AS summary, i.intent AS intent, toString(i.recorded_at) AS recorded_at, t.id AS ticket_id "
        "ORDER BY i.recorded_at DESC LIMIT 8",
        customer_id=customer_id,
    )
    session = records(
        "MATCH (:Customer {id: $customer_id})-[:HAS_ACTIVE_SESSION]->(s:SupportSession {id: $session_key}) "
        "RETURN s[$intent_key] AS intent, s[$product_key] AS product, s[$issue_key] AS issue_type, "
        "s[$question_key] AS question, toString(s[$updated_key]) AS updated_at",
        customer_id=customer_id,
        session_key=f"{customer_id}:{session_id}",
        intent_key="pending_intent",
        product_key="pending_product",
        issue_key="pending_issue_type",
        question_key="last_question",
        updated_key="updated_at",
    )
    messages = records(
        "MATCH (:Conversation {id: $session_key})-[:HAS_MESSAGE]->(message:Message) "
        "RETURN message.role AS role, message.content AS content, toString(message.created_at) AS created_at "
        "ORDER BY message.created_at DESC LIMIT 8",
        session_key=f"{customer_id}:{session_id}",
    )
    facts = records(
        "MATCH (:Customer {id: $customer_id})-[relationship]->(fact:Fact) "
        "WHERE type(relationship) = 'HAS_FACT' "
        "RETURN fact[$predicate_key] AS predicate, fact[$object_key] AS object, toString(fact[$updated_key]) AS updated_at "
        "ORDER BY fact[$updated_key] DESC LIMIT 6",
        customer_id=customer_id,
        predicate_key="predicate",
        object_key="object",
        updated_key="updated_at",
    )
    traces = records(
        "MATCH (:Customer {id: $customer_id})-[:HAS_REASONING_TRACE]->(trace:ReasoningTrace) "
        "OPTIONAL MATCH (trace)-[:HAS_STEP]->(step:ReasoningStep) "
        "RETURN trace.id AS id, trace.task AS task, trace.status AS status, trace.outcome AS outcome, trace.started_at AS started_at, "
        "collect({action: step[$action_key], intent: step[$intent_key], target_product: step[$product_key], "
        "issue_type: step[$issue_key], intent_confidence: step[$intent_confidence_key], product_confidence: step[$product_confidence_key], "
        "issue_confidence: step[$issue_confidence_key], threshold: step[$threshold_key], needs_clarification: step[$clarification_key], "
        "reason: step[$reason_key], result_count: step[$result_count_key], ticket_id: step[$ticket_id_key], outcome: step[$outcome_key]}) AS steps "
        "ORDER BY started_at DESC LIMIT 4",
        customer_id=customer_id,
        action_key="action",
        intent_key="intent",
        product_key="target_product",
        issue_key="issue_type",
        intent_confidence_key="intent_confidence",
        product_confidence_key="product_confidence",
        issue_confidence_key="issue_confidence",
        threshold_key="threshold",
        clarification_key="needs_clarification",
        reason_key="reason",
        result_count_key="result_count",
        ticket_id_key="ticket_id",
        outcome_key="outcome",
    )
    return {
        "counts": counts,
        "statuses": statuses,
        "customer": customer[0] if customer else None,
        "tickets": active_tickets,
        "interactions": interactions,
        "session": session[0] if session else None,
        "messages": [record for record in reversed(messages)],
        "facts": facts,
        "traces": traces,
    }


def customers():
    return records(
        "MATCH (c:Customer)-[:OPENED]->(t:Ticket) "
        "WITH c, count(t) AS ticket_count, "
        "sum(CASE WHEN t.status IN ['Open', 'Pending Customer Response'] THEN 1 ELSE 0 END) AS active "
        "WHERE ticket_count > 1 "
        "RETURN c.id AS id, c.name AS name, ticket_count, active ORDER BY active DESC, ticket_count DESC LIMIT 20"
    )


def flow(classification):
    route = [
        "http_request",
        "extract",
        "begin_trace",
        "load_session",
        "vector_recall",
        "jev_decision",
        "confidence_gate",
    ]
    if classification["needs_clarification"]:
        return route + ["clarify", "complete_trace"]
    return route + ["retrieve", "mutate", "openrouter_llm", "complete_trace"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        return

    def send_json(self, value, status=200):
        body = json.dumps(value, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/customers":
            return self.send_json(customers())
        if parsed.path == "/api/overview":
            params = parse_qs(parsed.query)
            customer_id = params.get("customer_id", [None])[0]
            session_id = params.get("session_id", ["demo"])[0]
            if not customer_id:
                options = customers()
                customer_id = options[0]["id"] if options else None
            if not customer_id:
                return self.send_json({"error": "No imported customers found."}, 404)
            return self.send_json(overview(customer_id, session_id))
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/chat":
            return self.send_json({"error": "Not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            customer_id = data["customer_id"]
            session_id = data.get("session_id", "demo")
            message = data["message"].strip()
            if not message:
                raise ValueError("Enter a message.")
            result = agent.invoke({"customer_id": customer_id, "session_id": session_id, "message": message})
            self.send_json({
                "response": result["response"],
                "classification": result["classification"],
                "ticket_id": result.get("ticket_id"),
                "flow": flow(result["classification"]),
                "overview": overview(customer_id, session_id),
            })
        except (KeyError, TypeError, ValueError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            self.send_json({"error": str(error)}, 500)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Demo UI: http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()