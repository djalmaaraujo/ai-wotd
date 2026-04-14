from datetime import date
from pathlib import Path

from wotd.corpus import (
    RawItem,
    article_id_for,
    build_day_stats,
    make_snippet,
    write_article_derivative,
)


def _raw(idx: int = 0) -> RawItem:
    return RawItem(
        source_id="fixture",
        external_id=f"guid-{idx}",
        url=f"https://example.com/post-{idx}",
        url_canonical=f"https://example.com/post-{idx}",
        title=f"Post about MCP #{idx}",
        author="Alice",
        published_at="2026-04-13T12:00:00+00:00",
        content_text=(
            "MCP is now supported across the platform. "
            "Agents can use MCP to discover tools. "
            "Context window stays at 1M tokens. " * 3
        ),
        kind="article",
    )


def test_article_id_is_deterministic():
    assert article_id_for("a", "b") == article_id_for("a", "b")
    assert article_id_for("a", "b") != article_id_for("a", "c")


def test_bucket_by_published_at_when_plausible(tmp_path: Path):
    """Sitemap backfill with old published_at → bucket by that date."""
    from datetime import date, datetime
    articles = tmp_path / "articles"
    cache = tmp_path / "cache"
    item = RawItem(
        source_id="beehiiv",
        external_id="guid-old",
        url="https://example.com/p/old",
        url_canonical="https://example.com/p/old",
        title="An old post",
        author="X",
        published_at="2025-08-15T10:00:00+00:00",  # ~8 months ago
        content_text="Historical content about MCP agents.",
        kind="article",
    )
    simulated_now = datetime(2026, 4, 14, 12, 0, 0)
    _, derivative = write_article_derivative(item, articles, cache, now=simulated_now)

    # Bucket goes to publication date, not the fetch date.
    assert (articles / "2025-08-15" / f"{derivative['article_id']}.json").exists()
    assert not (articles / "2026-04-14").exists()


def test_bucket_falls_back_to_fetched_at_on_missing_published(tmp_path: Path):
    from datetime import datetime
    articles = tmp_path / "articles"
    cache = tmp_path / "cache"
    item = RawItem(
        source_id="src",
        external_id="g",
        url="https://x/",
        url_canonical="https://x/",
        title="t",
        author=None,
        published_at="",  # missing
        content_text="body text",
        kind="article",
    )
    simulated_now = datetime(2026, 4, 14, 12, 0, 0)
    _, derivative = write_article_derivative(item, articles, cache, now=simulated_now)
    assert (articles / "2026-04-14" / f"{derivative['article_id']}.json").exists()


def test_bucket_rejects_future_dated_articles(tmp_path: Path):
    """Protects against feeds that publish with clock skew / far-future dates."""
    from datetime import datetime
    articles = tmp_path / "articles"
    cache = tmp_path / "cache"
    item = RawItem(
        source_id="src",
        external_id="g",
        url="https://x/",
        url_canonical="https://x/",
        title="Time traveller",
        author=None,
        published_at="2030-01-01T00:00:00+00:00",
        content_text="body",
        kind="article",
    )
    simulated_now = datetime(2026, 4, 14, 12, 0, 0)
    _, derivative = write_article_derivative(item, articles, cache, now=simulated_now)
    assert (articles / "2026-04-14" / f"{derivative['article_id']}.json").exists()


def test_make_snippet_caps_length():
    text = "x" * 500
    snip = make_snippet(text, max_chars=280)
    assert len(snip) <= 281  # allow the trailing ellipsis


def test_write_article_derivative_has_no_full_text(tmp_path: Path):
    articles = tmp_path / "articles"
    cache = tmp_path / "cache"
    item = _raw(0)
    out_path, derivative = write_article_derivative(item, articles, cache)

    assert out_path.exists()
    # Derivative must NOT contain the full text.
    assert "content_text" not in derivative
    assert "content_html" not in derivative
    assert derivative["content_chars"] == len(item.content_text)
    assert len(derivative["snippet"]) <= 281
    # Full text lives in the cache.
    cache_file = cache / f"{derivative['article_id']}.txt"
    assert cache_file.exists()
    assert cache_file.read_text() == item.content_text


def test_build_day_stats_from_cache(tmp_path: Path):
    from datetime import datetime
    articles = tmp_path / "articles"
    cache = tmp_path / "cache"
    stats = tmp_path / "stats"
    day = date(2026, 4, 13)
    simulated_now = datetime.combine(day, datetime.min.time())
    for i in range(3):
        write_article_derivative(_raw(i), articles, cache, now=simulated_now)
    summary = build_day_stats(articles, stats, cache, day)
    assert summary is not None
    assert summary["document_count"] == 3
    assert "mcp" in summary["terms"]
    assert summary["terms"]["mcp"]["df"] == 3
    assert (stats / f"{day.isoformat()}.json").exists()
