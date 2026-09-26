"""LangGraph customer support agent; Neo4j is the persistent memory."""
import json
import sys
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from clients import CHAT_MODEL, chat, embed, query, require_driver
from classifier import CONFIDENCE_THRESHOLD, classify
from extractor import extract_candidates


class State(TypedDict, total=False):
    customer_id: str
    session_id: str
    session_key: str
    conversation_id: str
    user_message_id: str
    trace_id: str
    query_embedding: list[float]
    message: str
    candidates: list[str]
    purchased_products: list[str]
    session: dict
    short_term_memory: list[dict]
    long_term_memory: list[dict]
    reasoning_memory: list[dict]
    semantic_memory: list[dict]
    classification: dict
    history: list[dict]
    ticket_id: str | None
    response: str


def extract(state):
    candidates = extract_candidates(state["message"])
    products = query(
        "MATCH (c:Customer {id: $id})-[:PURCHASED]->(p:Product) RETURN p.name AS name ORDER BY name",
        id=state["customer_id"],
    )
    purchased = [record["name"] for record in products]
    text = state["message"].casefold()
    candidates.extend(product for product in purchased if product.casefold() in text)
    session_id = state.get("session_id", state["customer_id"])
    return {
        "candidates": list(dict.fromkeys(candidates)),
        "purchased_products": purchased,
        "session_id": session_id,
        "session_key": f"{state['customer_id']}:{session_id}",
    }


def begin_trace(state):
    conversation_id = state["session_key"]
    user_message_id = str(uuid4())
    trace_id = str(uuid4())
    query_embedding = embed(state["message"])
    query(
        "MATCH (c:Customer {id: $customer_id}) "
        "MERGE (conversation:Conversation {id: $conversation_id}) "
        "ON CREATE SET conversation.session_id = $session_id, conversation.created_at = datetime() "
        "SET conversation.updated_at = datetime() "
        "MERGE (c)-[:HAS_CONVERSATION]->(conversation) "
        "OPTIONAL MATCH (conversation)-[last:LAST_MESSAGE]->(previous:Message) "
        "FOREACH (_ IN CASE WHEN last IS NULL THEN [] ELSE [1] END | DELETE last) "
        "CREATE (message:Message {id: $message_id, role: 'user', content: $content, embedding: $embedding, created_at: datetime()}) "
        "MERGE (conversation)-[:HAS_MESSAGE]->(message) "
        "MERGE (conversation)-[:LAST_MESSAGE]->(message) "
        "FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END | CREATE (previous)-[:NEXT_MESSAGE]->(message)) "
        "CREATE (trace:ReasoningTrace {id: $trace_id, task: $task, started_at: datetime(), status: 'running'}) "
        "MERGE (c)-[:HAS_REASONING_TRACE]->(trace) "
        "MERGE (trace)-[:TRIGGERED_BY]->(message) "
        "CREATE (step:ReasoningStep {id: $start_step_id, action: 'received_message', created_at: datetime()}) "
        "MERGE (trace)-[:HAS_STEP]->(step)",
        customer_id=state["customer_id"],
        conversation_id=conversation_id,
        session_id=state["session_id"],
        message_id=user_message_id,
        content=state["message"],
        embedding=query_embedding,
        trace_id=trace_id,
        task=state["message"][:200],
        start_step_id=str(uuid4()),
    )
    return {
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "trace_id": trace_id,
        "query_embedding": query_embedding,
    }


def load_session(state):
    records = query(
        "MATCH (c:Customer {id: $customer_id})-[:HAS_ACTIVE_SESSION]->(s:SupportSession {id: $session_key}) "
        "RETURN s[$intent_key] AS pending_intent, s[$product_key] AS pending_product, "
        "s[$issue_key] AS pending_issue_type LIMIT 1",
        customer_id=state["customer_id"],
        session_key=state["session_key"],
        intent_key="pending_intent",
        product_key="pending_product",
        issue_key="pending_issue_type",
    )
    return {"session": records[0].data() if records else {}}


def load_memory(state):
    messages = query(
        "MATCH (:Conversation {id: $conversation_id})-[:HAS_MESSAGE]->(message:Message) "
        "WHERE message.id <> $message_id "
        "RETURN message.role AS role, message.content AS content "
        "ORDER BY message.created_at DESC LIMIT 6",
        conversation_id=state["conversation_id"],
        message_id=state["user_message_id"],
    )
    facts = query(
        "MATCH (:Customer {id: $customer_id})-[relationship]->(fact:Fact) "
        "WHERE type(relationship) = 'HAS_FACT' "
        "RETURN fact[$subject_key] AS subject, fact[$predicate_key] AS predicate, fact[$object_key] AS object "
        "ORDER BY fact[$updated_key] DESC LIMIT 6",
        customer_id=state["customer_id"],
        subject_key="subject",
        predicate_key="predicate",
        object_key="object",
        updated_key="updated_at",
    )
    decisions = query(
        "MATCH (:Customer {id: $customer_id})-[:HAS_REASONING_TRACE]->(trace:ReasoningTrace)-[:HAS_STEP]->(step:ReasoningStep) "
        "WHERE trace.id <> $trace_id AND step[$action_key] = 'classify_with_jev' "
        "RETURN step[$intent_key] AS intent, step[$product_key] AS product, step[$issue_key] AS issue_type, "
        "step[$reason_key] AS reason "
        "ORDER BY trace[$started_key] DESC LIMIT 4",
        customer_id=state["customer_id"],
        trace_id=state["trace_id"],
        action_key="action",
        intent_key="intent",
        product_key="target_product",
        issue_key="issue_type",
        reason_key="reason",
        started_key="started_at",
    )
    try:
        semantic_messages = query(
            "CALL db.index.vector.queryNodes('message_embedding', 4, $embedding) YIELD node, score "
            "MATCH (:Customer {id: $customer_id})-[:HAS_CONVERSATION]->(:Conversation)-[:HAS_MESSAGE]->(node) "
            "WHERE node.id <> $message_id "
            "RETURN 'message' AS kind, node.content AS content, score ORDER BY score DESC",
            customer_id=state["customer_id"],
            message_id=state["user_message_id"],
            embedding=state["query_embedding"],
        )
        semantic_facts = query(
            "CALL db.index.vector.queryNodes('fact_embedding', 4, $embedding) YIELD node, score "
            "MATCH (:Customer {id: $customer_id})-[relationship]->(node) "
            "WHERE type(relationship) = 'HAS_FACT' "
            "RETURN 'fact' AS kind, node.object AS content, score ORDER BY score DESC",
            customer_id=state["customer_id"],
            embedding=state["query_embedding"],
        )
        semantic_memory = [record.data() for record in [*semantic_messages, *semantic_facts]]
    except Exception:
        semantic_memory = []
    return {
        "short_term_memory": [record.data() for record in reversed(messages)],
        "long_term_memory": [record.data() for record in facts],
        "reasoning_memory": [record.data() for record in decisions],
        "semantic_memory": semantic_memory,
    }


def classify_node(state):
    result = classify(
        state["message"],
        state["candidates"],
        state["purchased_products"],
        pending=state["session"],
    )
    return {"classification": result.model_dump()}


def record_decision(state):
    decision = state["classification"]
    reason = (
        "clarification_required: confidence below threshold"
        if decision["needs_clarification"]
        else "confidence threshold met"
    )
    query(
        "MATCH (trace:ReasoningTrace {id: $trace_id}) "
        "CREATE (step:ReasoningStep {id: $step_id, action: 'classify_with_jev', intent: $intent, "
        "target_product: $product, issue_type: $issue_type, intent_confidence: $intent_confidence, "
        "product_confidence: $product_confidence, issue_confidence: $issue_confidence, "
        "threshold: $threshold, needs_clarification: $needs_clarification, reason: $reason, created_at: datetime()}) "
        "MERGE (trace)-[:HAS_STEP]->(step) "
        "WITH step OPTIONAL MATCH (product:Product {name: $product}) "
        "FOREACH (_ IN CASE WHEN product IS NULL THEN [] ELSE [1] END | MERGE (step)-[:CLASSIFIED_PRODUCT]->(product))",
        trace_id=state["trace_id"],
        step_id=str(uuid4()),
        intent=decision["intent"],
        product=decision["target_product"],
        issue_type=decision["issue_type"],
        intent_confidence=decision["intent_confidence"],
        product_confidence=decision["product_confidence"],
        issue_confidence=decision["issue_confidence"],
        threshold=CONFIDENCE_THRESHOLD,
        needs_clarification=decision["needs_clarification"],
        reason=reason,
    )
    return {}


def retrieve(state):
    classification = state["classification"]
    if classification["intent"] not in {"FOLLOW_UP", "ESCALATION"}:
        return {"history": []}
    records = query(
        "MATCH (c:Customer {id: $id})-[:OPENED]->(t:Ticket) "
        "WHERE t.status IN ['Open', 'Pending Customer Response'] "
        "OPTIONAL MATCH (t)-[:REGARDS]->(p:Product) "
        "WITH c, t, p WHERE $product IS NULL OR p.name = $product "
        "OPTIONAL MATCH (c)-[:HAD_INTERACTION]->(i:Interaction)-[:UPDATES]->(t) "
        "WITH t, p, collect(i{.summary, .intent, .recorded_at}) AS history "
        "RETURN t.id AS ticket_id, t.status AS status, t.subject AS subject, "
        "t.type AS type, p.name AS product, history "
        "ORDER BY t.date_of_purchase DESC LIMIT 5",
        id=state["customer_id"],
        product=classification["target_product"],
    )
    return {"history": [record.data() for record in records]}


def record_retrieval(state):
    ticket_ids = [item["ticket_id"] for item in state["history"]]
    query(
        "MATCH (trace:ReasoningTrace {id: $trace_id}) "
        "CREATE (step:ReasoningStep {id: $step_id, action: 'retrieve_ticket_memory', result_count: $result_count, created_at: datetime()}) "
        "MERGE (trace)-[:HAS_STEP]->(step) "
        "WITH step, $ticket_ids AS ticket_ids "
        "UNWIND CASE WHEN size(ticket_ids) = 0 THEN [null] ELSE ticket_ids END AS ticket_id "
        "OPTIONAL MATCH (ticket:Ticket {id: ticket_id}) "
        "FOREACH (_ IN CASE WHEN ticket IS NULL THEN [] ELSE [1] END | MERGE (step)-[:USED_MEMORY]->(ticket))",
        trace_id=state["trace_id"],
        step_id=str(uuid4()),
        ticket_ids=ticket_ids,
        result_count=len(ticket_ids),
    )
    return {}


def mutate(state):
    customer_id = state["customer_id"]
    message = state["message"]
    decision = state["classification"]
    intent = decision["intent"]
    history = state["history"]
    ticket_id = history[0]["ticket_id"] if history else None
    if intent == "CHITCHAT":
        return {"ticket_id": None}
    if intent == "NEW_ISSUE" or (intent in {"FOLLOW_UP", "ESCALATION"} and not ticket_id):
        ticket_id = f"SUP-{uuid4().hex[:12]}"
        query(
            "MATCH (c:Customer {id: $customer_id}) "
            "CREATE (t:Ticket {id: $ticket_id, status: 'Open', subject: $summary, type: $intent, "
            "created_at: datetime()}) "
            "CREATE (c)-[:OPENED]->(t) "
            "WITH c, t OPTIONAL MATCH (c)-[:PURCHASED]->(p:Product {name: $product}) "
            "FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | CREATE (t)-[:REGARDS]->(p))",
            customer_id=customer_id,
            ticket_id=ticket_id,
            summary=message[:180],
            intent=intent,
            product=decision["target_product"],
        )

    interaction_id = str(uuid4())
    query(
        "MATCH (c:Customer {id: $customer_id}) "
        "CREATE (i:Interaction {id: $interaction_id, summary: $summary, intent: $intent, "
        "recorded_at: datetime()}) "
        "CREATE (c)-[:HAD_INTERACTION]->(i) "
        "WITH i OPTIONAL MATCH (t:Ticket {id: $ticket_id}) "
        "FOREACH (_ IN CASE WHEN t IS NULL THEN [] ELSE [1] END | CREATE (i)-[:UPDATES]->(t)) "
        "WITH t WHERE t IS NOT NULL AND $resolved = true SET t.status = 'Closed'",
        customer_id=customer_id,
        interaction_id=interaction_id,
        summary=message[:280],
        intent=intent,
        ticket_id=ticket_id,
        resolved=decision["is_resolved"],
    )
    if state.get("session_key"):
        query(
            "MATCH (:Customer {id: $customer_id})-[:HAS_ACTIVE_SESSION]->(s:SupportSession {id: $session_key}) "
            "DETACH DELETE s",
            customer_id=customer_id,
            session_key=state["session_key"],
        )
    return {"ticket_id": ticket_id}


def clarify(state):
    decision = state["classification"]
    query(
        "MATCH (c:Customer {id: $customer_id}) "
        "MERGE (s:SupportSession {id: $session_key}) "
        "ON CREATE SET s.customer_id = $customer_id, s.session_id = $session_id, s.created_at = datetime() "
        "SET s.pending_intent = $pending_intent, s.pending_product = $pending_product, "
        "s.pending_issue_type = $pending_issue_type, s.updated_at = datetime(), "
        "s.last_question = $last_question "
        "MERGE (c)-[:HAS_ACTIVE_SESSION]->(s)",
        customer_id=state["customer_id"],
        session_id=state["session_id"],
        session_key=state["session_key"],
        pending_intent=(
            decision["intent"]
            if decision["intent_confidence"] >= CONFIDENCE_THRESHOLD
            else state["session"].get("pending_intent")
        ),
        pending_product=(
            decision["target_product"]
            if decision["product_confidence"] >= CONFIDENCE_THRESHOLD
            else state["session"].get("pending_product")
        ),
        pending_issue_type=(
            decision["issue_type"]
            if decision["issue_confidence"] >= CONFIDENCE_THRESHOLD
            else state["session"].get("pending_issue_type")
        ),
        last_question=decision["clarification_question"],
    )
    return {"response": state["classification"]["clarification_question"]}


def generate(state):
    if chat is None:
        raise RuntimeError("Set OPENROUTER_API_KEY in .env before generating a response.")
    decision = state["classification"]
    prompt = (
        "You are a customer support agent. Use only the supplied graph context and current message. "
        "Acknowledge relevant prior interactions, do not ask for information already present, and give "
        "one practical next step. The imported ticket descriptions may be synthetic; rely on their subject, "
        "product, status, and category instead. If the user says the issue is fixed, acknowledge that and "
        "do not offer another troubleshooting step. Do not invent policies or purchase details.\n\n"
        f"Customer products: {json.dumps(state['purchased_products'], ensure_ascii=True, default=str)}\n"
        f"Short-term conversation memory: {json.dumps(state['short_term_memory'], ensure_ascii=True, default=str)}\n"
        f"Long-term facts: {json.dumps(state['long_term_memory'], ensure_ascii=True, default=str)}\n"
        f"Prior decision summaries: {json.dumps(state['reasoning_memory'], ensure_ascii=True, default=str)}\n"
        f"Semantic memory matches: {json.dumps(state['semantic_memory'], ensure_ascii=True, default=str)}\n"
        f"Retrieved ticket context: {json.dumps(state['history'], ensure_ascii=True, default=str)}\n"
        f"Current decision: {json.dumps(decision, default=str)}\n"
        f"Current message: {state['message']}"
    )
    result = chat.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return {"response": result.choices[0].message.content or "I couldn't prepare a response just now."}


def complete_trace(state):
    decision = state["classification"]
    outcome = "clarification_requested" if decision["needs_clarification"] else "response_generated"
    fact_id = f"{state.get('ticket_id')}:{decision['issue_type']}"
    response_embedding = embed(state["response"])
    fact_embedding = embed(f"{decision['target_product'] or 'unspecified'} | {decision['issue_type']}") if state.get("ticket_id") and decision["issue_type"] else None
    query(
        "MATCH (c:Customer {id: $customer_id})-[:HAS_CONVERSATION]->(conversation:Conversation {id: $conversation_id}) "
        "MATCH (trace:ReasoningTrace {id: $trace_id}) "
        "OPTIONAL MATCH (conversation)-[last:LAST_MESSAGE]->(previous:Message) "
        "FOREACH (_ IN CASE WHEN last IS NULL THEN [] ELSE [1] END | DELETE last) "
        "CREATE (message:Message {id: $message_id, role: 'assistant', content: $content, embedding: $response_embedding, created_at: datetime()}) "
        "MERGE (conversation)-[:HAS_MESSAGE]->(message) "
        "MERGE (conversation)-[:LAST_MESSAGE]->(message) "
        "FOREACH (_ IN CASE WHEN previous IS NULL THEN [] ELSE [1] END | CREATE (previous)-[:NEXT_MESSAGE]->(message)) "
        "MERGE (trace)-[:PRODUCED]->(message) "
        "CREATE (step:ReasoningStep {id: $step_id, action: 'complete_turn', outcome: $outcome, "
        "ticket_id: $ticket_id, created_at: datetime()}) "
        "MERGE (trace)-[:HAS_STEP]->(step) "
        "SET trace.status = 'complete', trace.outcome = $outcome, trace.success = true, trace.completed_at = datetime() "
        "WITH c, trace, $ticket_id AS ticket_id "
        "OPTIONAL MATCH (ticket:Ticket {id: ticket_id}) "
        "FOREACH (_ IN CASE WHEN ticket IS NULL OR $issue_type IS NULL THEN [] ELSE [1] END | "
        "MERGE (fact:Fact {id: $fact_id}) "
        "SET fact.subject = $customer_id, fact.predicate = 'reported_issue', "
        "fact.object = coalesce($product, 'unspecified') + ' | ' + $issue_type, fact.embedding = $fact_embedding, fact.updated_at = datetime() "
        "MERGE (c)-[:HAS_FACT]->(fact) "
        "MERGE (ticket)-[:EVIDENCES]->(fact) "
        "MERGE (trace)-[:COMMITTED_FACT]->(fact))",
        customer_id=state["customer_id"],
        conversation_id=state["conversation_id"],
        trace_id=state["trace_id"],
        message_id=str(uuid4()),
        content=state["response"],
        response_embedding=response_embedding,
        step_id=str(uuid4()),
        outcome=outcome,
        ticket_id=state.get("ticket_id"),
        issue_type=decision["issue_type"],
        product=decision["target_product"],
        fact_id=fact_id,
        fact_embedding=fact_embedding,
    )
    return {}


graph_builder = StateGraph(State)
graph_builder.add_node("extract", extract)
graph_builder.add_node("begin_trace", begin_trace)
graph_builder.add_node("load_session", load_session)
graph_builder.add_node("load_memory", load_memory)
graph_builder.add_node("classify", classify_node)
graph_builder.add_node("record_decision", record_decision)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("record_retrieval", record_retrieval)
graph_builder.add_node("mutate", mutate)
graph_builder.add_node("clarify", clarify)
graph_builder.add_node("generate", generate)
graph_builder.add_node("complete_trace", complete_trace)
graph_builder.add_edge(START, "extract")
graph_builder.add_edge("extract", "begin_trace")
graph_builder.add_edge("begin_trace", "load_session")
graph_builder.add_edge("load_session", "load_memory")
graph_builder.add_edge("load_memory", "classify")
graph_builder.add_edge("classify", "record_decision")
graph_builder.add_conditional_edges(
    "record_decision",
    lambda state: "clarify" if state["classification"]["needs_clarification"] else "retrieve",
)
graph_builder.add_edge("clarify", "complete_trace")
graph_builder.add_edge("retrieve", "record_retrieval")
graph_builder.add_edge("record_retrieval", "mutate")
graph_builder.add_edge("mutate", "generate")
graph_builder.add_edge("generate", "complete_trace")
graph_builder.add_edge("complete_trace", END)
agent = graph_builder.compile()


def main():
    if chat is None:
        raise SystemExit("Set OPENROUTER_API_KEY in .env before starting the agent.")
    customer_id = sys.argv[1] if len(sys.argv) > 1 else None
    session_id = sys.argv[2] if len(sys.argv) > 2 else None
    if not customer_id:
        repeat = query(
            "MATCH (c:Customer)-[:OPENED]->(t:Ticket) "
            "WITH c, count(t) AS tickets, "
            "sum(CASE WHEN t.status IN ['Open', 'Pending Customer Response'] THEN 1 ELSE 0 END) AS active "
            "WHERE tickets > 1 RETURN c.id AS id, tickets ORDER BY active DESC, tickets DESC LIMIT 1"
        )
        if not repeat:
            raise SystemExit("No repeat customer found; run `python import_dataset.py` first.")
        customer_id = repeat[0]["id"]
        print(f"Using repeat-customer sample: {customer_id}")
    try:
        while True:
            message = input("You> ").strip()
            if message.lower() in {"quit", "exit"}:
                break
            if message:
                result = agent.invoke({
                    "customer_id": customer_id,
                    "session_id": session_id or customer_id,
                    "message": message,
                })
                print(f"Agent> {result['response']}\n")
    finally:
        require_driver().close()


if __name__ == "__main__":
    main()