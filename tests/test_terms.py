from collections import Counter

from wotd.terms import (
    build_candidate_pool,
    extract_terms,
    ngrams,
    summarize_per_day,
    title_terms,
    tokenize,
    top_terms,
    trim_edges,
)


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


def test_extract_terms_rejects_single_char_tokens_in_ngrams():
    """PDF-leakage regression: "n n", "q q", "w w" must NOT survive."""
    # Looks like PDF stream residue.
    text = "n n q q w w obj endobj endstream n n"
    counts = extract_terms(text)
    assert "n n" not in counts
    assert "q q" not in counts
    assert "w w" not in counts
    # Single-letter unigrams also out.
    assert "n" not in counts
    assert "q" not in counts
    # And the PDF artifacts are stopworded.
    assert "obj" not in counts
    assert "endobj" not in counts


def test_extract_terms_still_keeps_normal_ngrams():
    text = "The context window for this model is one million tokens."
    counts = extract_terms(text)
    # Allowlisted multi-word term survives.
    assert counts.get("context window", 0) >= 1
    # Good unigrams survive.
    assert counts.get("tokens", 0) >= 1
    assert counts.get("model", 0) >= 1


def test_tokenize_keeps_version_numbers():
    toks = tokenize("GPT-5.6 beats Claude 4.5 on SWE-bench.")
    assert "gpt-5.6" in toks
    assert "4.5" not in toks
    assert "swe-bench" in toks


def test_tokenize_splits_domains_instead_of_swallowing_them():
    """The dot is for versions; a domain must not become one token."""
    toks = tokenize("Read claude.com and simonwillison.net today.Tomorrow too")
    assert "claude.com" not in toks
    assert "claude" in toks and "com" in toks
    assert "simonwillison.net" not in toks
    assert "today.tomorrow" not in toks


def test_trim_edges_drops_leading_and_trailing_filler():
    assert trim_edges("the gemini") == "gemini"
    assert trim_edges("ai deepseek-v4 preview") == "deepseek-v4 preview"
    assert trim_edges("fable is back") == "fable is back"
    assert trim_edges("of the") == ""


def test_trim_edges_keeps_ai_when_stripping_it_leaves_one_word():
    """'ai safety' is a term; 'safety' is not the same thing."""
    assert trim_edges("ai safety") == "ai safety"
    assert trim_edges("ai act") == "ai act"
    assert trim_edges("new relic") == "new relic"


def test_title_terms_needs_two_titles():
    titles = ["Claude Tag ships today", "Claude Tag lands in Slack", "Gemini ships"]
    found = title_terms(titles)
    assert "claude tag" in found
    assert "gemini" not in found  # only one title mentions it


def test_title_terms_counts_headlines_not_repetitions():
    assert title_terms(["Sora 2 beats Sora 1 in the Sora era"]) == []


def test_build_candidate_pool_merges_trims_and_drops_subphrases():
    pool = build_candidate_pool(
        ["the gemini", "he breaks down", "claude"],
        ["Claude Tag ships", "Claude Tag lands"],
    )
    assert "gemini" in pool
    assert "claude tag" in pool
    assert "claude" not in pool  # sub-phrase of "claude tag"
    assert "the gemini" not in pool


def test_build_candidate_pool_keeps_a_name_hidden_inside_a_fragment():
    """A fragment must not suppress the clean name it contains."""
    pool = build_candidate_pool(["access to fable", "fable"], [])
    assert "fable" in pool
    assert "access to fable" in pool


def test_build_candidate_pool_keeps_a_name_a_noisy_bigram_contains():
    """'adoption claude' must not delete 'claude' before the judge sees it."""
    pool = build_candidate_pool(["adoption claude", "claude"], [])
    assert "claude" in pool


def test_build_candidate_pool_respects_the_cap():
    body = [f"term{i}" for i in range(80)]
    assert len(build_candidate_pool(body, [], cap=25)) == 25
