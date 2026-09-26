"""Fast local candidate extraction for support messages."""
import spacy


_nlp = None


def extract_candidates(text):
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    doc = _nlp(text)
    return list(dict.fromkeys(
        span.text.strip()
        for span in [*doc.noun_chunks, *doc.ents]
        if span.text.strip()
    ))[:20]


if __name__ == "__main__":
    candidates = extract_candidates("My GoPro Hero access stopped working last week.")
    assert any("GoPro Hero" in item for item in candidates), candidates
    assert any("last week" in item for item in candidates), candidates
    print(candidates)