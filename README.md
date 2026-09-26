# Graph-memory support agent

Copy `.env.example` values into the ignored `.env` file and set `OPENROUTER_API_KEY`.
Both the classifier and the reply generator route through OpenRouter with that key:
`JEV_MODEL` (default `typesafe/jev-1.13`) makes the structured decision, `CHAT_MODEL`
(default `openai/gpt-4o-mini`) generates the customer-facing reply — Jev's endpoint
only outputs structured decisions, not free text, so a separate chat model still
drafts the reply from Jev's decision plus graph context.

```sh
.venv/bin/python import_dataset.py
.venv/bin/python test_classifier.py
.venv/bin/python -m unittest -v
.venv/bin/python agent.py
.venv/bin/python app.py
```

Open `http://127.0.0.1:8000` for the demo UI. It runs the same LangGraph agent
as the CLI and shows its persisted Neo4j tickets, interactions, and clarification session.

The importer is idempotent and adds the 8,469 ticket rows as `Customer`, `Product`,
`Ticket`, and `Interaction` nodes using the plan's relationships. The source has
139 repeat customers, three real statuses, and 42 products. Its descriptions are
synthetic/noisy, so the agent uses ticket subjects for memory summaries and does not
send raw descriptions to the response model.

Jev scores the message against controlled intent and issue labels, then scores
products only from the customer's recorded purchases, self-reporting a 0-1 confidence
per field. A new issue or escalation with any required score below
`CLASSIFICATION_CONFIDENCE_THRESHOLD` (default `0.85`) is routed to a deterministic
clarification response; it never queries, creates, or updates ticket memory. The
clarification state is instead saved as a customer-owned `SupportSession`, retaining
only high-confidence intent, product, and issue slots. The next message merges these
verified slots with its classification. Once a ticket is created or updated, the
session is removed. Follow-ups retrieve only matching open ticket memory, while
high-confidence new issues create a ticket and interaction before reply generation.
Pass an optional second argument to `agent.py` to isolate concurrent sessions for the
same customer: `python agent.py <customer-id> <session-id>`.

The LangGraph routing, confidence gate, and mutation behavior are covered by offline
unit tests (`test_agent.py`, mocking the Jev call). `test_classifier.py` is a live
smoke test against the real Jev endpoint and requires imported data plus a working
`OPENROUTER_API_KEY`.
