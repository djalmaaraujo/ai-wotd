import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from wotd.corpus import RawItem, build_day_stats, write_article_derivative
from wotd.export import export_all
from wotd.data import articles, terms, trending, wotd


def _build_fixture_corpus(root: Path) -> None:
    articles_dir = root / "articles"
    cache_dir = root / "cache"
    stats_dir = root / "stats"
    wotd_dir = root / "wotd"
    parquet_dir = root / "parquet"

    # Two days of data. Pass `now=` explicitly so the fetched_at-based
    # bucket lands on the simulated day, not on today.
    today = date(2026, 4, 13)
    yesterday = today - timedelta(days=1)
    for d, suffix in [(yesterday, "y"), (today, "t")]:
        simulated_now = datetime.combine(d, datetime.min.time())
        for i in range(2):
            item = RawItem(
                source_id="src",
                external_id=f"{suffix}-{i}",
                url=f"https://example.com/{suffix}/{i}",
                url_canonical=f"https://example.com/{suffix}/{i}",
                title=f"Post MCP {i}",
                author="Alice",
                published_at=f"{d.isoformat()}T12:00:00+00:00",
                content_text="MCP " + "agents " * (i + 2) + "context window",
                kind="article",
            )
            write_article_derivative(
                item, articles_dir, cache_dir, now=simulated_now
            )
        build_day_stats(articles_dir, stats_dir, cache_dir, d)

    # Write a wotd entry for today manually.
    wotd_dir.mkdir(parents=True, exist_ok=True)
    (wotd_dir / f"{today.isoformat()}.json").write_text(
        json.dumps(
            {
                "date": today.isoformat(),
                "word": "mcp",
                "score": 3.14,
                "candidates": [{"term": "mcp"}, {"term": "agents"}],
                "evidence_article_ids": [],
            }
        )
    )

    export_all(articles_dir, stats_dir, wotd_dir, parquet_dir)


def test_data_api_articles_and_terms(tmp_path: Path):
    _build_fixture_corpus(tmp_path)
    pq = tmp_path / "parquet"

    arts = articles(parquet_dir=pq)
    assert isinstance(arts, pl.DataFrame)
    assert arts.height == 4

    mcp_terms = terms(term="mcp", parquet_dir=pq)
    assert mcp_terms.height == 2  # one row per day

    top = trending(n=5, since="30d", parquet_dir=pq)
    assert top.height >= 1
    assert "mcp" in top["term"].to_list()

    days = wotd(parquet_dir=pq)
    assert days.height == 1
    assert days[0, "word"] == "mcp"
