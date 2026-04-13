"""Tokenization, n-gram extraction, per-article and per-day term stats."""

from __future__ import annotations

import re
from collections import Counter
from importlib import resources
from pathlib import Path


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*[A-Za-z0-9]|[A-Za-z]")


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
) -> Counter:
    """Return a Counter of terms (unigrams..n-grams) with stopword filtering.

    A term is kept when:
      * it's in the allowlist, OR
      * it has at least one non-stopword token and ≥2 chars and not all digits.
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
            # Drop pure-stopword grams.
            if all(p in stopwords for p in parts):
                continue
            # Require at least one alpha-bearing content token.
            if all(p.isdigit() for p in parts):
                continue
            if n == 1:
                tok = parts[0]
                if tok in stopwords or len(tok) < 2 or tok.isdigit():
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
