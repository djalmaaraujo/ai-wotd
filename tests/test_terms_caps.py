"""Regression tests for the per-day stats size caps."""

from __future__ import annotations

from collections import Counter

from wotd.terms import (
    MAX_ARTICLES_PER_TERM,
    MAX_AUTHORS_PER_TERM,
    summarize_per_day,
)


def test_summarize_drops_df_one_terms():
    per_article = {
        "a1": Counter({"mcp": 2, "only_once": 1}),
        "a2": Counter({"mcp": 3}),
    }
    day = summarize_per_day(per_article)
    assert "mcp" in day["terms"]
    # Single-doc terms are dropped to keep stats small.
    assert "only_once" not in day["terms"]


def test_summarize_caps_article_list_per_term():
    per_article = {
        f"a{i}": Counter({"mcp": 1}) for i in range(500)
    }
    day = summarize_per_day(per_article, max_articles_per_term=20)
    mcp = day["terms"]["mcp"]
    assert mcp["df"] == 500  # df still correct
    assert len(mcp["articles"]) == 20  # but list is capped


def test_summarize_caps_authors_per_term():
    per_article = {
        f"a{i}": Counter({"mcp": 1}) for i in range(50)
    }
    authors = {f"a{i}": f"author-{i}" for i in range(50)}
    day = summarize_per_day(per_article, authors, max_authors_per_term=5)
    assert len(day["terms"]["mcp"]["authors"]) == 5


def test_summarize_caps_total_term_count():
    # 10 unique df=2 terms; cap the output to the top 3 by tf.
    per_article = {
        "a1": Counter({f"t{i}": (10 - i) for i in range(10)}),
        "a2": Counter({f"t{i}": (10 - i) for i in range(10)}),
    }
    day = summarize_per_day(per_article, max_terms=3)
    assert len(day["terms"]) == 3
    # t0, t1, t2 are the highest-tf → should be the survivors.
    assert set(day["terms"].keys()) == {"t0", "t1", "t2"}


def test_defaults_are_sane():
    assert MAX_ARTICLES_PER_TERM >= 20
    assert MAX_AUTHORS_PER_TERM >= 5
