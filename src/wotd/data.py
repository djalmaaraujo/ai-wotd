"""Public Python API for the ai-wotd corpus.

Thin wrapper over DuckDB + Parquet that returns Polars DataFrames. Use from a
checked-out copy of the repo:

    from wotd.data import articles, terms, wotd, trending
    trending(n=10, since="7d")

All functions accept `parquet_dir` to point at a different mirror of the data.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl


DEFAULT_PARQUET_DIR = Path("data/parquet")

_RELATIVE_RE = re.compile(r"^(\d+)d$")


def _resolve_since(since: str | date | datetime | None) -> date | None:
    if since is None:
        return None
    if isinstance(since, datetime):
        return since.date()
    if isinstance(since, date):
        return since
    if not isinstance(since, str):
        raise TypeError(f"since: unexpected type {type(since).__name__}")
    m = _RELATIVE_RE.match(since.strip())
    if m:
        return date.today() - timedelta(days=int(m.group(1)))
    # ISO date.
    return date.fromisoformat(since)


def _conn(parquet_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    parquet_dir = Path(parquet_dir)
    for name in ("articles", "terms", "wotd"):
        path = parquet_dir / f"{name}.parquet"
        if path.exists():
            # read_parquet() doesn't accept prepared parameters, so embed
            # the path directly with escaped single quotes.
            escaped = str(path).replace("'", "''")
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{escaped}')"
            )
    return con


def _to_polars(rel: "duckdb.DuckDBPyRelation") -> pl.DataFrame:
    return pl.from_arrow(rel.arrow())  # type: ignore[return-value]


def articles(
    *,
    source: str | Iterable[str] | None = None,
    author: str | None = None,
    since: str | date | datetime | None = None,
    until: str | date | datetime | None = None,
    kind: str | None = None,
    parquet_dir: str | Path = DEFAULT_PARQUET_DIR,
) -> pl.DataFrame:
    """Return a Polars DataFrame of articles, filtered."""
    con = _conn(Path(parquet_dir))
    clauses: list[str] = []
    params: list = []
    if source is not None:
        if isinstance(source, str):
            clauses.append("source_id = ?")
            params.append(source)
        else:
            sources = list(source)
            placeholders = ",".join(["?"] * len(sources))
            clauses.append(f"source_id IN ({placeholders})")
            params.extend(sources)
    if author:
        clauses.append("author = ?")
        params.append(author)
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    since_d = _resolve_since(since)
    if since_d:
        clauses.append("CAST(published_at AS DATE) >= CAST(? AS DATE)")
        params.append(since_d.isoformat())
    until_d = _resolve_since(until)
    if until_d:
        clauses.append("CAST(published_at AS DATE) <= CAST(? AS DATE)")
        params.append(until_d.isoformat())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM articles {where} ORDER BY published_at DESC"
    return _to_polars(con.execute(sql, params))


def terms(
    *,
    term: str | None = None,
    since: str | date | datetime | None = None,
    until: str | date | datetime | None = None,
    parquet_dir: str | Path = DEFAULT_PARQUET_DIR,
) -> pl.DataFrame:
    con = _conn(Path(parquet_dir))
    clauses: list[str] = []
    params: list = []
    if term:
        clauses.append("term = ?")
        params.append(term)
    since_d = _resolve_since(since)
    if since_d:
        clauses.append("CAST(date AS DATE) >= CAST(? AS DATE)")
        params.append(since_d.isoformat())
    until_d = _resolve_since(until)
    if until_d:
        clauses.append("CAST(date AS DATE) <= CAST(? AS DATE)")
        params.append(until_d.isoformat())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM terms {where} ORDER BY date DESC, tf DESC, term"
    return _to_polars(con.execute(sql, params))


def wotd(
    *,
    on: str | date | datetime | None = None,
    since: str | date | datetime | None = None,
    parquet_dir: str | Path = DEFAULT_PARQUET_DIR,
) -> pl.DataFrame:
    """Without arguments, returns every day of WOTD, newest first."""
    con = _conn(Path(parquet_dir))
    clauses: list[str] = []
    params: list = []
    if on is not None:
        d = _resolve_since(on)
        clauses.append("date = ?")
        params.append(d.isoformat())  # type: ignore[union-attr]
    since_d = _resolve_since(since)
    if since_d:
        clauses.append("CAST(date AS DATE) >= CAST(? AS DATE)")
        params.append(since_d.isoformat())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM wotd {where} ORDER BY date DESC"
    return _to_polars(con.execute(sql, params))


def trending(
    n: int = 10,
    since: str | date | datetime | None = "7d",
    parquet_dir: str | Path = DEFAULT_PARQUET_DIR,
) -> pl.DataFrame:
    """Top N terms by total tf over the window."""
    con = _conn(Path(parquet_dir))
    since_d = _resolve_since(since)
    clauses = []
    params: list = []
    if since_d:
        clauses.append("CAST(date AS DATE) >= CAST(? AS DATE)")
        params.append(since_d.isoformat())
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT term,
               SUM(tf) AS total_tf,
               SUM(df) AS total_df,
               COUNT(DISTINCT date) AS days
        FROM terms
        {where}
        GROUP BY term
        ORDER BY total_tf DESC, term
        LIMIT {int(n)}
    """
    return _to_polars(con.execute(sql, params))
