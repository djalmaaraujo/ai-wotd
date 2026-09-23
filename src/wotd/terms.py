"""Tokenization, n-gram extraction, per-article and per-day term stats."""

from __future__ import annotations

import re
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Iterable


# The dot keeps model versions whole ("gpt-5.6"); without it they collapse to
# their major number and can never be elected.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-\.]*[A-Za-z0-9]|[A-Za-z]")


def _read_resource(name: str) -> list[str]:
    path = resources.files("wotd.resources").joinpath(name)
    with path.open("r", encoding="utf-8") as f:
        return [
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def load_stopwords() -> frozenset[str]:
    return frozenset(_read_resource("stopwords.txt"))


def load_allowlist() -> frozenset[str]:
    return frozenset(_read_resource("ai_terms_allowlist.txt"))


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; keep hyphenated compounds and apostrophes."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def extract_terms(
    text: str,
    *,
    stopwords: frozenset[str] | None = None,
    allowlist: frozenset[str] | None = None,
    max_n: int = 3,
    min_token_len: int = 2,
) -> Counter:
    """Return a Counter of terms (unigrams..n-grams) with stopword filtering.

    A term is kept when:
      * it's in the allowlist, OR
      * every token is at least `min_token_len` chars, has a non-digit
        component, and the gram isn't entirely stopwords.
    """
    stopwords = stopwords if stopwords is not None else load_stopwords()
    allowlist = allowlist if allowlist is not None else load_allowlist()

    tokens = tokenize(text)
    counts: Counter = Counter()

    for n in range(1, max_n + 1):
        for gram in ngrams(tokens, n):
            if gram in allowlist:
                counts[gram] += 1
                continue
            parts = gram.split(" ")
            # Reject grams where any individual token is too short or all-digit.
            # This is what traps "n n", "q q", "obj obj", etc. from PDF leakage.
            if any(len(p) < min_token_len for p in parts):
                continue
            if any(p.isdigit() for p in parts):
                continue
            # Drop pure-stopword grams.
            if all(p in stopwords for p in parts):
                continue
            if n == 1:
                tok = parts[0]
                if tok in stopwords:
                    continue
            counts[gram] += 1

    return counts


MAX_TERMS_PER_DAY = 5000
MAX_ARTICLES_PER_TERM = 50
MAX_AUTHORS_PER_TERM = 10


def summarize_per_day(
    per_article_counts: dict[str, Counter],
    article_authors: dict[str, str] | None = None,
    *,
    max_terms: int = MAX_TERMS_PER_DAY,
    max_articles_per_term: int = MAX_ARTICLES_PER_TERM,
    max_authors_per_term: int = MAX_AUTHORS_PER_TERM,
) -> dict:
    """Collapse per-article term counters into a per-day stats blob.

    Defense-in-depth against pathological days that contain hundreds of
    articles and blow the per-term article list into megabytes:

      * Drop terms with df=1 (can never win WOTD — the scorer requires
        df >= 2 anyway) so the long tail of single-doc n-grams is culled.
      * Cap term article lists at `max_articles_per_term` (default 50).
      * Cap author lists at `max_authors_per_term` (default 10).
      * Keep only the top `max_terms` terms by tf (default 5000).

    Returns a dict with:
      terms: { term: { tf, df, articles: [...], authors: [...] } }
      document_count: int
      article_ids: [...]
    """
    article_authors = article_authors or {}
    terms: dict[str, dict] = {}
    for article_id, counter in per_article_counts.items():
        for term, tf in counter.items():
            t = terms.setdefault(
                term, {"tf": 0, "df": 0, "articles": [], "authors": []}
            )
            t["tf"] += tf
            t["df"] += 1
            if len(t["articles"]) < max_articles_per_term:
                t["articles"].append(article_id)
            author = article_authors.get(article_id)
            if (
                author
                and author not in t["authors"]
                and len(t["authors"]) < max_authors_per_term
            ):
                t["authors"].append(author)

    # Drop single-doc terms (can't be elected; just noise on disk) ONLY
    # when the day has enough mass for df>=2 to be reachable. On slow days
    # with a single article, keep everything so the scorer still has
    # something to work with.
    if len(per_article_counts) >= 2:
        terms = {k: v for k, v in terms.items() if v["df"] >= 2}

    # Keep top-N by tf (then alphabetical for stable output).
    if len(terms) > max_terms:
        top = sorted(
            terms.items(), key=lambda kv: (-kv[1]["tf"], kv[0])
        )[:max_terms]
        terms = dict(top)

    # Stable ordering inside each term entry.
    for v in terms.values():
        v["articles"].sort()
        v["authors"].sort()

    return {
        "terms": terms,
        "document_count": len(per_article_counts),
        "article_ids": sorted(per_article_counts.keys()),
    }


def top_terms(counter: Counter, n: int = 20) -> list[tuple[str, int]]:
    """Deterministic top-N: (-count, term) sort."""
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


EDGE_WORDS = frozenset(
    {
        "ai", "new", "how", "why", "what", "this", "that", "it", "its",
        "the", "a", "an", "to", "of", "in", "is", "and", "for", "with", "on", "at",
    }
)

POOL_CAP = 60
TITLE_MIN_COUNT = 2


def _has_interior_filler(term: str, edges: frozenset[str]) -> bool:
    parts = term.split()
    return any(part in edges for part in parts[1:-1])


def trim_edges(term: str, stopwords: frozenset[str] | None = None) -> str:
    """Strip filler words from both ends of a candidate term."""
    stopwords = stopwords if stopwords is not None else load_stopwords()
    edges = stopwords | EDGE_WORDS
    parts = term.split()
    while parts and parts[0] in edges:
        parts = parts[1:]
    while parts and parts[-1] in edges:
        parts = parts[:-1]
    return " ".join(parts)


def title_terms(
    titles: Iterable[str],
    *,
    min_count: int = TITLE_MIN_COUNT,
    stopwords: frozenset[str] | None = None,
    allowlist: frozenset[str] | None = None,
) -> list[str]:
    """Terms that several of today's headlines share, most frequent first.

    Headlines carry the day's topic without the navigation chrome, subscribe
    prompts and PDF debris that pollute article bodies.
    """
    stopwords = stopwords if stopwords is not None else load_stopwords()
    allowlist = allowlist if allowlist is not None else load_allowlist()
    counts: Counter = Counter()
    for title in titles:
        if title:
            counts.update(
                extract_terms(title, stopwords=stopwords, allowlist=allowlist)
            )
    return [term for term, count in top_terms(counts, n=300) if count >= min_count]


def build_candidate_pool(
    body_terms: Iterable[str],
    titles: Iterable[str],
    *,
    cap: int = POOL_CAP,
    stopwords: frozenset[str] | None = None,
    allowlist: frozenset[str] | None = None,
) -> list[str]:
    """Merge body and headline candidates into one clean, deduplicated pool."""
    stopwords = stopwords if stopwords is not None else load_stopwords()
    allowlist = allowlist if allowlist is not None else load_allowlist()

    merged: list[str] = list(body_terms)
    for term in title_terms(titles, stopwords=stopwords, allowlist=allowlist):
        if term not in merged:
            merged.append(term)

    ordered: dict[str, None] = {}
    for term in merged:
        trimmed = trim_edges(term, stopwords=stopwords)
        if len(trimmed) < 2 or trimmed.replace(".", "").isdigit():
            continue
        ordered.setdefault(trimmed, None)

    kept = list(ordered)
    edges = stopwords | EDGE_WORDS
    redundant = {
        short
        for short in kept
        for long in kept
        if short != long
        and f" {short} " in f" {long} "
        and not _has_interior_filler(long, edges)
    }
    return [term for term in kept if term not in redundant][:cap]
