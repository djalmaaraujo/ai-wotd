"""Tokenization, n-gram extraction, per-article and per-day term stats."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable


# The dot is allowed only before digits, so model versions stay whole
# ("gpt-5.6") while domains still split into their parts.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*(?:\.[0-9]+)*|[A-Za-z]|[0-9]+(?:\.[0-9]+)*")
_HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-"})


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


@dataclass(frozen=True)
class _Token:
    text: str
    number: bool
    named: bool
    joined: bool


def _scan(text: str) -> list[_Token]:
    """Tokens, each marking whether only whitespace separates it from the one before."""
    if not text:
        return []
    text = text.translate(_HYPHENS)
    tokens = []
    end = 0
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        number = raw[0].isdigit()
        named = not number and (raw[0].isupper() or any(c.isdigit() for c in raw))
        joined = not text[end : match.start()].strip()
        tokens.append(_Token(raw.lower(), number, named, joined))
        end = match.end()
    return tokens


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; keep hyphenated compounds, apostrophes and version numbers."""
    return [token.text for token in _scan(text)]


def ngrams(tokens: list[str], n: int) -> list[str]:
    if n <= 0 or len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _numbers_name_a_version(window: list[_Token], stopwords: frozenset[str]) -> bool:
    """True when every number in the gram follows a capitalised or versioned name."""
    for k, token in enumerate(window):
        if not token.number:
            continue
        before = window[k - 1] if k else None
        if before is None or not before.named or before.text in stopwords:
            return False
    return True


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
      * every word is at least `min_token_len` chars, every number is the
        version of the name just before it ("opus 5.5"), and the gram isn't
        entirely stopwords.
    """
    stopwords = stopwords if stopwords is not None else load_stopwords()
    allowlist = allowlist if allowlist is not None else load_allowlist()

    tokens = _scan(text)
    counts: Counter = Counter()

    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]
            if not all(token.joined for token in window[1:]):
                continue
            parts = [token.text for token in window]
            gram = " ".join(parts)
            if gram in allowlist:
                counts[gram] += 1
                continue
            if not _numbers_name_a_version(window, stopwords):
                continue
            # Reject grams where any individual word is too short.
            # This is what traps "n n", "q q", "obj obj", etc. from PDF leakage.
            if any(len(t.text) < min_token_len for t in window if not t.number):
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
        "how", "why", "what", "this", "that", "it", "its",
        "the", "a", "an", "to", "of", "in", "is", "and", "for", "with", "on", "at",
        "introducing", "announcing", "meet",
    }
)

# "ai safety" and "new relic" are terms; "safety" and "relic" are not. These
# only come off when something more than a single word is left behind.
WEAK_EDGE_WORDS = frozenset({"ai", "new"})

POOL_CAP = 80
TITLE_MIN_COUNT = 2
NAME_MAX_WORDS = 5
NAMES_CAP = 25
TITLE_CASE_MIN_CAPITALISED = 2

_TITLE_TAG = re.compile(r"^\s*\[[^\]]*\]\s*")
_TITLE_SEPARATOR = re.compile(r"\s+(?:\||·|\\|—|–|-)\s+")


def _qualifies(short: str, long: str) -> bool:
    """True when `long` is `short` with qualifying words after it.

    "claude tag" is a fuller name for "claude", so the bare name can go. A
    phrase that only ends in the name, such as "adoption claude", is a
    different thing and must not delete the name it swallowed.
    """
    return long.startswith(f"{short} ")


def trim_edges(term: str, stopwords: frozenset[str] | None = None) -> str:
    """Strip filler words from both ends of a candidate term."""
    stopwords = stopwords if stopwords is not None else load_stopwords()
    edges = (stopwords | EDGE_WORDS) - WEAK_EDGE_WORDS
    parts = term.split()
    while parts and parts[0] in edges:
        parts = parts[1:]
    while parts and parts[-1] in edges:
        parts = parts[:-1]
    while len(parts) > 2 and parts[0] in WEAK_EDGE_WORDS:
        parts = parts[1:]
    while len(parts) > 2 and parts[-1] in WEAK_EDGE_WORDS:
        parts = parts[:-1]
    return " ".join(parts)


def clean_title(title: str | None) -> str:
    """The headline itself, without the site name or a leading "[tag]"."""
    title = _TITLE_TAG.sub("", title or "").strip()
    parts = [part.strip() for part in _TITLE_SEPARATOR.split(title) if part.strip()]
    if not parts:
        return ""
    return max(parts, key=lambda part: len(part.split()))


def _is_title_case(tokens: list[_Token], stopwords: frozenset[str]) -> bool:
    capitalised = [t for t in tokens[1:] if t.named and t.text in stopwords]
    return len(capitalised) >= TITLE_CASE_MIN_CAPITALISED


def _name_runs(tokens: list[_Token], stopwords: frozenset[str]) -> list[list[str]]:
    runs: list[list[str]] = []
    run: list[str] = []
    previous: _Token | None = None
    for token in tokens:
        if run and not token.joined:
            runs.append(run)
            run = []
        if token.number and run and previous is not None and not previous.number:
            run.append(token.text)
        elif token.named and len(token.text) > 1 and token.text not in stopwords:
            run.append(token.text)
        else:
            if run:
                runs.append(run)
            run = []
        previous = token
    if run:
        runs.append(run)
    return runs


def title_names(
    titles: Iterable[str],
    *,
    term_df: dict[str, int] | None = None,
    cap: int = NAMES_CAP,
    stopwords: frozenset[str] | None = None,
) -> list[str]:
    """Capitalised names in today's headlines ("Gemini Omni 1.1 Flash"), most cited first.

    Names are ranked by how many headlines carry them, then by how many of the
    day's documents use them (`term_df`; a name longer than the stats keep is
    looked up by its first three words).

    One headline is enough: a launch post names its product once, and the
    two-headline rule in `title_terms` would never see it. The first word of a
    headline is capitalised whether or not it is a name, so each name is also
    offered without it.
    """
    stopwords = stopwords if stopwords is not None else load_stopwords()
    counts: Counter = Counter()
    for title in titles:
        tokens = _scan(clean_title(title))
        if _is_title_case(tokens, stopwords):
            continue
        found: set[str] = set()
        for run in _name_runs(tokens, stopwords):
            if len(run) == 1 and tokens and run[0] == tokens[0].text and run[0].isalpha():
                continue
            for words in (run, run[1:]):
                name = trim_edges(" ".join(words[:NAME_MAX_WORDS]), stopwords)
                if name and not name.split()[0][0].isdigit():
                    found.add(name)
        counts.update(found)
    term_df = term_df or {}

    def rank(item: tuple[str, int]) -> tuple:
        name, headlines = item
        df = term_df.get(name) or term_df.get(" ".join(name.split()[:3]), 0)
        return (-headlines, -df, name)

    return [name for name, _ in sorted(counts.items(), key=rank)[:cap]]


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
                set(extract_terms(clean_title(title), stopwords=stopwords, allowlist=allowlist))
            )
    return [term for term, count in top_terms(counts, n=300) if count >= min_count]


def build_candidate_pool(
    body_terms: Iterable[str],
    titles: Iterable[str],
    *,
    term_df: dict[str, int] | None = None,
    cap: int = POOL_CAP,
    stopwords: frozenset[str] | None = None,
    allowlist: frozenset[str] | None = None,
) -> list[str]:
    """Merge body and headline candidates into one clean, deduplicated pool."""
    stopwords = stopwords if stopwords is not None else load_stopwords()
    allowlist = allowlist if allowlist is not None else load_allowlist()

    titles = list(titles)
    counted = _clean_terms(
        list(body_terms) + title_terms(titles, stopwords=stopwords, allowlist=allowlist),
        stopwords,
    )
    # A name from a single headline is a guess, so it may join the pool but
    # must not push out a shorter term the counts back.
    named = _clean_terms(title_names(titles, term_df=term_df, stopwords=stopwords), stopwords)

    edges = (stopwords | EDGE_WORDS) - WEAK_EDGE_WORDS
    qualifiers = [long for long in counted if not any(part in edges for part in long.split())]

    def redundant(short: str) -> bool:
        return any(short != long and _qualifies(short, long) for long in qualifiers)

    kept = [term for term in counted if not redundant(term)]
    extra = [term for term in named if term not in kept and not redundant(term)]
    return (kept + extra)[:cap]


def _clean_terms(terms: Iterable[str], stopwords: frozenset[str]) -> list[str]:
    ordered: dict[str, None] = {}
    for term in terms:
        trimmed = trim_edges(term, stopwords=stopwords)
        if len(trimmed) < 2 or trimmed.replace(".", "").isdigit():
            continue
        ordered.setdefault(trimmed, None)
    return list(ordered)
