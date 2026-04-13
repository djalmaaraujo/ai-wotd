"""Write data/parquet/{articles,terms,wotd}.parquet from the JSON corpus.

Idempotent; byte-stable given identical inputs (deterministic row order,
pinned pyarrow).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


_ARTICLE_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("source_id", pa.string()),
        ("via_source_id", pa.string()),
        ("kind", pa.string()),
        ("url", pa.string()),
        ("url_canonical", pa.string()),
        ("title", pa.string()),
        ("author", pa.string()),
        ("published_at", pa.string()),
        ("fetched_at", pa.string()),
        ("content_sha256", pa.string()),
        ("content_chars", pa.int64()),
        ("content_tokens", pa.int64()),
        ("snippet", pa.string()),
    ]
)


_TERM_SCHEMA = pa.schema(
    [
        ("date", pa.string()),
        ("term", pa.string()),
        ("tf", pa.int64()),
        ("df", pa.int64()),
        ("article_ids", pa.list_(pa.string())),
        ("authors", pa.list_(pa.string())),
    ]
)


_WOTD_SCHEMA = pa.schema(
    [
        ("date", pa.string()),
        ("word", pa.string()),
        ("score", pa.float64()),
        ("evidence_article_ids", pa.list_(pa.string())),
        ("candidate_terms", pa.list_(pa.string())),
        ("llm_summary", pa.string()),
        ("llm_why", pa.string()),
        ("llm_model", pa.string()),
        ("llm_generated_at", pa.string()),
    ]
)


def _iter_article_jsons(articles_dir: Path) -> Iterable[dict]:
    if not articles_dir.exists():
        return []
    for path in sorted(articles_dir.glob("*/*.json")):
        with open(path, "r", encoding="utf-8") as f:
            yield json.load(f)


def _iter_stats_jsons(stats_dir: Path) -> Iterable[dict]:
    if not stats_dir.exists():
        return []
    for path in sorted(stats_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            yield json.load(f)


def _iter_wotd_jsons(wotd_dir: Path) -> Iterable[dict]:
    if not wotd_dir.exists():
        return []
    for path in sorted(wotd_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            yield json.load(f)


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def export_articles(articles_dir: Path, out: Path) -> int:
    rows = []
    for a in _iter_article_jsons(articles_dir):
        rows.append(
            {
                "article_id": a.get("article_id"),
                "source_id": a.get("source_id"),
                "via_source_id": a.get("via_source_id"),
                "kind": a.get("kind"),
                "url": a.get("url"),
                "url_canonical": a.get("url_canonical"),
                "title": a.get("title"),
                "author": a.get("author"),
                "published_at": a.get("published_at"),
                "fetched_at": a.get("fetched_at"),
                "content_sha256": a.get("content_sha256"),
                "content_chars": int(a.get("content_chars") or 0),
                "content_tokens": int(a.get("content_tokens") or 0),
                "snippet": a.get("snippet"),
            }
        )
    rows.sort(key=lambda r: (r["published_at"] or "", r["article_id"] or ""))
    table = pa.Table.from_pylist(rows, schema=_ARTICLE_SCHEMA)
    _write_parquet(out, table)
    return len(rows)


def export_terms(stats_dir: Path, out: Path) -> int:
    rows = []
    for day in _iter_stats_jsons(stats_dir):
        d = day.get("date")
        for term, info in (day.get("terms") or {}).items():
            rows.append(
                {
                    "date": d,
                    "term": term,
                    "tf": int(info.get("tf") or 0),
                    "df": int(info.get("df") or 0),
                    "article_ids": list(info.get("articles") or []),
                    "authors": list(info.get("authors") or []),
                }
            )
    rows.sort(key=lambda r: (r["date"], r["term"]))
    table = pa.Table.from_pylist(rows, schema=_TERM_SCHEMA)
    _write_parquet(out, table)
    return len(rows)


def export_wotd(wotd_dir: Path, out: Path) -> int:
    rows = []
    for w in _iter_wotd_jsons(wotd_dir):
        llm = w.get("llm") or {}
        rows.append(
            {
                "date": w.get("date"),
                "word": w.get("word"),
                "score": float(w.get("score") or 0.0),
                "evidence_article_ids": list(w.get("evidence_article_ids") or []),
                "candidate_terms": [c.get("term") for c in (w.get("candidates") or [])],
                "llm_summary": llm.get("summary"),
                "llm_why": llm.get("why"),
                "llm_model": llm.get("model"),
                "llm_generated_at": llm.get("generated_at"),
            }
        )
    rows.sort(key=lambda r: r["date"] or "")
    table = pa.Table.from_pylist(rows, schema=_WOTD_SCHEMA)
    _write_parquet(out, table)
    return len(rows)


def export_all(
    articles_dir: Path, stats_dir: Path, wotd_dir: Path, parquet_dir: Path
) -> dict:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    return {
        "articles": export_articles(articles_dir, parquet_dir / "articles.parquet"),
        "terms": export_terms(stats_dir, parquet_dir / "terms.parquet"),
        "wotd": export_wotd(wotd_dir, parquet_dir / "wotd.parquet"),
    }
