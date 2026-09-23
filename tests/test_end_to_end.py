"""End-to-end smoke test that bypasses network and exercises the full pipeline."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from wotd.config import Paths
from wotd.corpus import RawItem, build_day_stats, write_article_derivative
from wotd.export import export_all
from wotd.site import render_site
from wotd.wotd import pick_wotd


TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


def _raw(i: int) -> RawItem:
    return RawItem(
        source_id="fixture",
        external_id=f"g-{i}",
        url=f"https://example.com/p/{i}",
        url_canonical=f"https://example.com/p/{i}",
        title=f"MCP update {i}",
        author="Alice",
        published_at="2026-04-13T12:00:00+00:00",
        content_text=(
            "MCP, agents, context window, reasoning — all trending today. "
            "MCP MCP MCP MCP MCP MCP."
        ),
        kind="article",
    )


def test_end_to_end(tmp_path: Path):
    paths = Paths.from_root(tmp_path)
    paths.ensure()

    # 1. Write fake articles (no network). Pin the bucket date explicitly.
    from datetime import date, datetime
    d = date(2026, 4, 13)
    simulated_now = datetime.combine(d, datetime.min.time())
    for i in range(3):
        write_article_derivative(
            _raw(i), paths.articles, paths.fulltext_cache, now=simulated_now
        )

    # 2. Process stats.
    summary = build_day_stats(paths.articles, paths.stats, paths.fulltext_cache, d)
    assert summary is not None

    # 3. Pick WOTD.
    payload = pick_wotd(paths.stats, paths.wotd, d, baseline_days=7)
    assert payload is not None
    assert payload["word"] is not None

    # 4. Export parquet.
    counts = export_all(paths.articles, paths.stats, paths.wotd, paths.parquet)
    assert counts["articles"] == 3
    assert counts["wotd"] == 1
    assert (paths.parquet / "articles.parquet").exists()
    assert (paths.parquet / "terms.parquet").exists()
    assert (paths.parquet / "wotd.parquet").exists()

    # 5. Build site with a custom base path (simulates GH Pages project URL).
    render_site(
        wotd_dir=paths.wotd,
        articles_dir=paths.articles,
        templates_dir=TEMPLATES_DIR,
        docs_dir=paths.docs,
        site_base="/ai-wotd",
        site_url="https://example.github.io/ai-wotd",
    )
    index_html = (paths.docs / "index.html").read_text()
    assert "<h1>" in index_html
    assert payload["word"] in index_html
    # Base path should prefix every in-site link.
    assert 'href="/ai-wotd/style.css"' in index_html
    assert 'href="/ai-wotd/archive/"' in index_html
    assert (paths.docs / ".nojekyll").exists()
    assert (paths.docs / "archive" / "index.html").exists()
    assert (paths.docs / "d" / d.isoformat() / "index.html").exists()
    assert (paths.docs / "feed.xml").exists()
    feed = (paths.docs / "feed.xml").read_text()
    assert "https://example.github.io/ai-wotd/d/" in feed


def test_render_site_with_empty_base(tmp_path: Path):
    """Custom-domain / root deployment: base='' should produce `/` links."""
    paths = Paths.from_root(tmp_path)
    paths.ensure()
    render_site(
        wotd_dir=paths.wotd,
        articles_dir=paths.articles,
        templates_dir=TEMPLATES_DIR,
        docs_dir=paths.docs,
        site_base="",
        site_url="https://example.com",
    )
    index_html = (paths.docs / "index.html").read_text()
    assert 'href="/style.css"' in index_html
    assert 'href="/archive/"' in index_html


def test_site_renders_a_quiet_day(tmp_path):
    """An abstained day gets its own page instead of vanishing."""
    import json as _json
    from wotd.site import render_site

    wotd_dir = tmp_path / "wotd"
    wotd_dir.mkdir()
    (wotd_dir / "2026-04-22.json").write_text(
        _json.dumps(
            {
                "date": "2026-04-22",
                "word": None,
                "status": "quiet_day",
                "candidates": [{"term": "nitter net", "score": 1.0, "tf_today": 8,
                                "df_today": 2, "avg_tf_baseline": 0.0, "articles": []}],
                "evidence_article_ids": [],
                "judge": {
                    "status": "ok",
                    "signals": {"nitter net": {"specificity": 0.1, "dominance": 0.2,
                                               "novelty": 0.9, "form": 0.3, "event": 0.1}},
                },
            }
        )
    )

    render_site(
        wotd_dir=wotd_dir,
        articles_dir=tmp_path / "articles",
        templates_dir=Path(__file__).resolve().parents[1] / "templates",
        docs_dir=tmp_path / "docs",
    )

    page = (tmp_path / "docs" / "d" / "2026-04-22" / "index.html").read_text()
    assert "no word today" in page
    assert "nitter net" in page
    assert "no word" in (tmp_path / "docs" / "archive" / "index.html").read_text()
