from collections import Counter

from wotd.terms import extract_terms, ngrams, summarize_per_day, tokenize, top_terms


def test_tokenize_keeps_hyphenated_and_apostrophe():
    toks = tokenize("Open-source models and it's MCP time.")
    assert "open-source" in toks
    assert "it's" in toks
    assert "mcp" in toks


def test_ngrams():
    assert ngrams(["a", "b", "c"], 2) == ["a b", "b c"]
    assert ngrams(["a"], 2) == []


def test_extract_terms_filters_stopwords_but_keeps_allowlist():
    text = "The MCP protocol and the context window are trending."
    counts = extract_terms(text)
    # stopwords like 'the' and 'and' should be gone
    assert "the" not in counts
    assert "and" not in counts
    # allowlisted phrases should survive
    assert counts["mcp"] >= 1
    assert counts["context window"] >= 1


def test_summarize_per_day_rolls_up_tf_df_and_articles():
    per_article = {
        "a1": Counter({"mcp": 3, "agents": 1}),
        "a2": Counter({"mcp": 2}),
    }
    authors = {"a1": "alice", "a2": "bob"}
    day = summarize_per_day(per_article, authors)
    assert day["document_count"] == 2
    assert day["terms"]["mcp"]["tf"] == 5
    assert day["terms"]["mcp"]["df"] == 2
    assert day["terms"]["mcp"]["articles"] == ["a1", "a2"]
    assert day["terms"]["mcp"]["authors"] == ["alice", "bob"]


def test_top_terms_is_deterministic():
    c = Counter({"b": 2, "a": 2, "c": 1})
    # Same count → alphabetical.
    assert top_terms(c, n=2) == [("a", 2), ("b", 2)]
