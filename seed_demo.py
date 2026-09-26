"""Create a deterministic returning customer for the live demo."""
from clients import query


def main():
    query(
        "MERGE (customer:Customer {id: $customer_id}) "
        "SET customer.name = 'DEMO / Alex Morgan' "
        "MERGE (product:Product {id: $product_id}) "
        "SET product.name = $product_id "
        "MERGE (customer)-[:PURCHASED]->(product) "
        "UNWIND $tickets AS row "
        "MERGE (ticket:Ticket {id: row.id}) "
        "ON CREATE SET ticket.status = 'Closed', ticket.subject = row.subject, ticket.type = 'HISTORICAL', "
        "ticket.created_at = datetime() "
        "MERGE (customer)-[:OPENED]->(ticket) "
        "MERGE (ticket)-[:REGARDS]->(product)",
        customer_id="demo@relay.local",
        product_id="Relay Office Suite",
        tickets=[
            {"id": "DEMO-HISTORY-1", "subject": "Previous Relay Office Suite setup question"},
            {"id": "DEMO-HISTORY-2", "subject": "Previous Relay Office Suite billing question"},
        ],
    )
    print("Demo customer ready: DEMO / Alex Morgan")


if __name__ == "__main__":
    main()