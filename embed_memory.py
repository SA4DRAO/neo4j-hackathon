"""Backfill vector embeddings for persisted conversations and facts."""
from clients import embed, query


def backfill(cypher, write_cypher):
    rows = [record.data() for record in query(cypher)]
    if not rows:
        return 0
    for row in rows:
        row["embedding"] = embed(row["text"])
    query(write_cypher, rows=rows)
    return len(rows)


def main():
    messages = backfill(
        "MATCH (message:Message) WHERE message.embedding IS NULL "
        "RETURN message.id AS id, message.content AS text",
        "UNWIND $rows AS row MATCH (message:Message {id: row.id}) SET message.embedding = row.embedding",
    )
    facts = backfill(
        "MATCH (fact:Fact) WHERE fact.embedding IS NULL "
        "RETURN fact.id AS id, fact.object AS text",
        "UNWIND $rows AS row MATCH (fact:Fact {id: row.id}) SET fact.embedding = row.embedding",
    )
    print(f"Embedded {messages} messages and {facts} facts.")


if __name__ == "__main__":
    main()