from collections import Counter

from wotd.terms import (
    build_candidate_pool,
    clean_title,
    extract_terms,
    ngrams,
    summarize_per_day,
    surface_form,
    title_names,
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
    assert "4.5" in toks
    assert "swe-bench" in toks


def test_tokenize_reads_non_breaking_hyphens_as_hyphens():
    assert "gpt-6" in tokenize("Introducing GPT\u20116 Astra")


def test_extract_terms_keeps_the_version_on_a_named_release():
    counts = extract_terms("Google shipped Gemini 4 Argon. Claude Opus 5.5 and Fable 5 follow.")
    assert counts["gemini 4 argon"] == 1
    assert counts["opus 5.5"] == 1
    assert counts["claude opus 5.5"] == 1
    assert counts["fable 5"] == 1
    assert "gemini argon" not in counts


def test_extract_terms_never_makes_a_term_out_of_a_bare_number():
    counts = extract_terms("It costs 5.5 dollars and 4 cents, says the 2026 report.")
    assert "5.5" not in counts
    assert "costs 5.5" not in counts
    assert "4 cents" not in counts
    assert "2026 report" not in counts


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


def test_clean_title_drops_the_publisher_and_newsletter_tags():
    assert clean_title("OpenAI launches GPT-6.1 Sol, says it nearly matches GPT-6 Astra | TechCrunch") == (
        "OpenAI launches GPT-6.1 Sol, says it nearly matches GPT-6 Astra"
    )
    assert clean_title("GLM-5.3 and the spread of advanced cyber capabilities \\ Anthropic") == (
        "GLM-5.3 and the spread of advanced cyber capabilities"
    )
    assert clean_title("The politics of panic - by Jerusalem Demsas - The Argument") == "The politics of panic"
    assert clean_title("[AINews] Pi 1.0, Pi Durable, and AIE NYC") == "Pi 1.0, Pi Durable, and AIE NYC"
    assert clean_title("Text-to-Speech is here") == "Text-to-Speech is here"


def test_title_names_finds_the_product_a_single_headline_names():
    names = title_names(
        [
            "Gemini Omni 1.1 Flash lets you build with more control",
            "Introducing Claude Sonnet 5 | Anthropic",
            "Expanding OpenAI's presence in Brazil",
        ]
    )
    assert "gemini omni 1.1 flash" in names
    assert "claude sonnet 5" in names
    assert "introducing claude sonnet 5" not in names


def test_title_names_ignores_headlines_written_in_title_case():
    assert title_names(["The Agent Said It Was Done. The Database Disagreed."]) == []


def test_build_candidate_pool_includes_names_from_one_headline():
    pool = build_candidate_pool(["google", "models"], ["Gemini Omni 1.1 Flash lets you build with more control"])
    assert "gemini omni 1.1 flash" in pool


def test_title_names_skips_a_lone_capital_that_only_starts_the_headline():
    assert title_names(["Making AI cheaper", "Since the preview, GPT-6 got faster"]) == ["gpt-6"]


def test_extract_terms_does_not_join_words_across_punctuation():
    counts = extract_terms("Meet Opus, Sonnet and Haiku. Models ship today.")
    assert "opus sonnet" not in counts
    assert "haiku models" not in counts
    assert counts["sonnet"] == 1


def test_title_names_stop_at_punctuation():
    assert "gemini 4 argon" in title_names(["[AINews] Gemini 4 Argon: GDM's answer to Astra"])
    assert "gemini 4 argon gdm's" not in title_names(["[AINews] Gemini 4 Argon: GDM's answer to Astra"])


def test_a_headline_name_never_hides_a_shorter_candidate():
    pool = build_candidate_pool(["claude tag"], ["Anthropic ships Claude Tag Krea today"])
    assert "claude tag" in pool


def test_a_headline_name_is_dropped_when_a_counted_term_qualifies_it():
    pool = build_candidate_pool(["claude tag"], ["Claude ships everywhere"])
    assert "claude" not in pool


def test_surface_form_spells_the_term_the_way_the_articles_do():
    texts = [
        "OpenAI launches GPT-6.1 Sol",
        "GPT‑6.1 Sol is cheaper than gpt-6.1 sol pro",
        "Why GPT-6.1 Sol matters",
    ]
    assert surface_form("gpt-6.1 sol", texts) == "GPT-6.1 Sol"


def test_surface_form_keeps_the_term_when_no_article_spells_it():
    assert surface_form("claude tag", ["Nothing here", "claude-tagged"]) == "claude tag"
