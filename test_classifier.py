"""Small live Jev routing smoke test; requires imported data and OPENROUTER_API_KEY."""
from clients import query
from classifier import classify
from extractor import extract_candidates


product = "GoPro Hero"


def main():
    customers = query(
        "MATCH (c:Customer)-[:PURCHASED]->(:Product {name: $product}) "
        "RETURN c.id AS id LIMIT 1",
        product=product,
    )
    assert customers, f"No dataset customer purchased {product}"
    customer_id = customers[0]["id"]
    messages = [
        ("My GoPro Hero just stopped powering on. This is a new issue.", "NEW_ISSUE", False),
        ("The GoPro Hero still will not power on after the steps from my last ticket.", "FOLLOW_UP", False),
        ("I'm fed up with the recurring GoPro Hero failure and may cancel if this continues.", "ESCALATION", False),
        ("Does my GoPro Hero support 4K video?", "CHITCHAT", False),
        ("The GoPro Hero works now, thanks. The issue is fixed.", "FOLLOW_UP", True),
    ]

    for message, expected_intent, expected_resolved in messages:
        result = classify(message, extract_candidates(message), [product])
        assert result.intent == expected_intent, (message, result, expected_intent)
        assert result.is_resolved is expected_resolved, (message, result, expected_resolved)
        assert result.target_product == product, (message, result)
        print(result.model_dump())

    print(f"Jev routing checks passed for {customer_id}")


if __name__ == "__main__":
    main()