"""Per-article derivative JSON + per-day stats orchestration.

The committed article JSON is derivative-only: metadata + snippet + stats +
hashes. Full text lives in the Actions cache (`Paths.fulltext_cache`) for the
duration of a run and is never committed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from .terms import extract_terms, load_allowlist, load_stopwords, summarize_per_day, top_terms

logger = logging.getLogger(__name__)


SNIPPET_MAX_CHARS = 280
# ~4 chars/token is the common rule-of-thumb without tokenizing.
TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4


@dataclass
class RawItem:
    source_id: str
    external_id: str            # feed GUID or URL
    url: str
    url_canonical: str
    title: str
    author: str | None
    published_at: str           # ISO 8601 UTC
    content_text: str           # full text — NEVER committed
    kind: str                   # "article" | "tweet"
    via_source_id: str | None = None


def article_id_for(source_id: str, external_id: str) -> str:
    h = hashlib.sha256(f"{source_id}\x00{external_id}".encode("utf-8")).hexdigest()[:8]
    return f"{source_id}--{h}"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // TOKEN_ESTIMATE_CHARS_PER_TOKEN)


def make_snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Prefer a word boundary.
    space = cut.rfind(" ")
    if space > max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:-") + "…"


def parse_pub_date(published_at: str) -> date:
    try:
        return datetime.fromisoformat(published_at.replace("Z", "+00:00")).date()
    except Exception:
        return date.today()


def write_article_derivative(
    item: RawItem,
    articles_dir: Path,
    fulltext_cache_dir: Path,
    *,
    stopwords: frozenset[str] | None = None,
    allowlist: frozenset[str] | None = None,
    now: datetime | None = None,
) -> tuple[Path, dict]:
    """Persist the derivative JSON and cache the full text separately.

    Returns (path_to_derivative_json, derivative_dict).
    """
    now = now or datetime.utcnow()
    stopwords = stopwords if stopwords is not None else load_stopwords()
    allowlist = allowlist if allowlist is not None else load_allowlist()

    article_id = article_id_for(item.source_id, item.external_id)
    pub_date = parse_pub_date(item.published_at)

    full_text = item.content_text or ""
    content_sha256 = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

    # Derive per-article term counts for the day-rollup step. We use the title
    # plus the full body so that headline terms get amplified once.
    combined = (item.title or "") + "\n" + full_text
    counts = extract_terms(combined, stopwords=stopwords, allowlist=allowlist)

    derivative = {
        "article_id": article_id,
        "source_id": item.source_id,
        "via_source_id": item.via_source_id,
        "kind": item.kind,
        "url": item.url,
        "url_canonical": item.url_canonical,
        "title": item.title,
        "author": item.author,
        "published_at": item.published_at,
        "fetched_at": now.replace(microsecond=0).isoformat() + "Z",
        "content_sha256": content_sha256,
        "content_chars": len(full_text),
        "content_tokens": estimate_tokens(full_text),
        "snippet": make_snippet(full_text if item.kind != "tweet" else full_text),
        "top_terms": [list(t) for t in top_terms(counts, n=20)],
    }

    day_dir = articles_dir / pub_date.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    out_path = day_dir / f"{article_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(derivative, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    # Cache full text (never committed — see .gitignore).
    fulltext_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = fulltext_cache_dir / f"{article_id}.txt"
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return out_path, derivative


def load_full_text(article_id: str, fulltext_cache_dir: Path) -> str | None:
    path = fulltext_cache_dir / f"{article_id}.txt"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def iter_articles_on_day(articles_dir: Path, d: date) -> Iterable[dict]:
    day_dir = articles_dir / d.isoformat()
    if not day_dir.exists():
        return []
    for path in sorted(day_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            yield json.load(f)


def build_day_stats(
    articles_dir: Path,
    stats_dir: Path,
    fulltext_cache_dir: Path,
    d: date,
    *,
    stopwords: frozenset[str] | None = None,
    allowlist: frozenset[str] | None = None,
) -> dict | None:
    """Combine the day's article derivatives into a per-day stats file."""
    stopwords = stopwords if stopwords is not None else load_stopwords()
    allowlist = allowlist if allowlist is not None else load_allowlist()

    per_article: dict[str, Counter] = {}
    authors: dict[str, str] = {}

    articles = list(iter_articles_on_day(articles_dir, d))
    if not articles:
        return None

    for derivative in articles:
        article_id = derivative["article_id"]
        full_text = load_full_text(article_id, fulltext_cache_dir) or ""
        if full_text:
            combined = (derivative.get("title") or "") + "\n" + full_text
            counts = extract_terms(combined, stopwords=stopwords, allowlist=allowlist)
        else:
            # Fall back to committed top_terms (accurate counts) — reprocess
            # paths without the cache still work on existing data.
            counts = Counter()
            for term, tf in derivative.get("top_terms", []):
                counts[term] = int(tf)
        per_article[article_id] = counts
        if derivative.get("author"):
            authors[article_id] = derivative["author"]

    summary = summarize_per_day(per_article, article_authors=authors)
    summary["date"] = d.isoformat()

    stats_dir.mkdir(parents=True, exist_ok=True)
    out_path = stats_dir / f"{d.isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return summary
