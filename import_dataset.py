"""Idempotently import the Kaggle tickets into the Aura graph."""
import csv
import sys
from datetime import datetime, timezone

from clients import DB, query, require_driver


def value(row, key):
    text = row[key].strip()
    return text or None


def rows(path):
    imported_at = datetime.now(timezone.utc).isoformat()
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            customer_id = value(row, "Customer Email")
            product = value(row, "Product Purchased")
            ticket_id = value(row, "Ticket ID")
            if not customer_id or not product or not ticket_id:
                continue
            yield {
                "customer_id": customer_id,
                "customer_name": value(row, "Customer Name"),
                "customer_age": int(row["Customer Age"]) if row["Customer Age"].strip() else None,
                "customer_gender": value(row, "Customer Gender"),
                "product_id": product,
                "ticket_id": ticket_id,
                "status": value(row, "Ticket Status"),
                "ticket_type": value(row, "Ticket Type"),
                "subject": value(row, "Ticket Subject"),
                "description": value(row, "Ticket Description"),
                "priority": value(row, "Ticket Priority"),
                "channel": value(row, "Ticket Channel"),
                "date_of_purchase": value(row, "Date of Purchase"),
                "resolution": value(row, "Resolution"),
                "first_response_time": value(row, "First Response Time"),
                "time_to_resolution": value(row, "Time to Resolution"),
                "satisfaction_rating": float(row["Customer Satisfaction Rating"])
                if row["Customer Satisfaction Rating"].strip() else None,
                "imported_at": imported_at,
            }


IMPORT = """
UNWIND $rows AS row
MERGE (c:Customer {id: row.customer_id})
SET c.name = row.customer_name, c.age = row.customer_age, c.gender = row.customer_gender
MERGE (p:Product {id: row.product_id})
SET p.name = row.product_id
MERGE (t:Ticket {id: row.ticket_id})
SET t.status = row.status, t.type = row.ticket_type, t.subject = row.subject,
    t.description = row.description, t.priority = row.priority, t.channel = row.channel,
    t.date_of_purchase = date(row.date_of_purchase), t.resolution = row.resolution,
    t.first_response_time = row.first_response_time, t.time_to_resolution = row.time_to_resolution,
    t.satisfaction_rating = row.satisfaction_rating
MERGE (c)-[:PURCHASED]->(p)
MERGE (c)-[:OPENED]->(t)
MERGE (t)-[:REGARDS]->(p)
MERGE (i:Interaction {id: row.ticket_id + '-source'})
SET i.summary = row.subject, i.intent = row.ticket_type, i.recorded_at = row.imported_at
MERGE (c)-[:HAD_INTERACTION]->(i)
MERGE (i)-[:UPDATES]->(t)
"""


def main(path="data/customer_support_tickets.csv"):
    driver = require_driver()
    with open("schema.cypher", encoding="utf-8") as source:
        statements = [
            "\n".join(line for line in part.splitlines() if not line.lstrip().startswith("//")).strip()
            for part in source.read().split(";")
        ]
    for statement in statements:
        if statement:
            driver.execute_query(statement, database_=DB)

    count = 0
    batch = []
    for row in rows(path):
        batch.append(row)
        if len(batch) == 250:
            driver.execute_query(IMPORT, {"rows": batch}, database_=DB)
            count += len(batch)
            batch.clear()
    if batch:
        driver.execute_query(IMPORT, {"rows": batch}, database_=DB)
        count += len(batch)
    stats = query(
        "MATCH (c:Customer) WITH count(c) AS customers "
        "MATCH (p:Product) WITH customers, count(p) AS products "
        "MATCH (t:Ticket) RETURN customers, products, count(t) AS tickets"
    )[0]
    print(f"Imported/upserted {count} tickets; graph has {stats['customers']} customers, "
          f"{stats['products']} products, {stats['tickets']} tickets.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/customer_support_tickets.csv")