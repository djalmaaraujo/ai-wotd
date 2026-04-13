"""Tests for the archive-page scraper adapter."""

from __future__ import annotations

import httpx
import pytest

from wotd.sources.archive import (
    ArchiveAdapter,
    _extract_issue_urls,
    _extract_published_from_meta,
)
from wotd.sources.base import Cursor


ARCHIVE_HTML = """
<!doctype html>
<html><body>
  <nav><a href="/subscribe">Subscribe</a></nav>
  <main>
    <a href="/p/issue-42-agents-are-eating-the-world">Issue 42</a>
    <a href="/p/issue-41-mcp-edition">Issue 41</a>
    <a href="https://codenewsletter.ai/p/issue-41-mcp-edition?utm_source=x">Issue 41 dup</a>
    <a href="/p/issue-40-rlhf">Issue 40</a>
    <a href="https://twitter.com/codenewsletter">Twitter</a>
    <a href="/about">About</a>
  </main>
</body></html>
"""


def test_extract_issue_urls_filters_to_prefix_and_host():
    urls = _extract_issue_urls(
        ARCHIVE_HTML,
        base_url="https://codenewsletter.ai/archive",
        issue_path_prefix="/p/",
    )
    # Twitter + /about + /subscribe are excluded.
    # The duplicate (same canonical) appears once.
    assert len(urls) == 3
    assert all("/p/" in u for u in urls)
    assert not any("twitter" in u for u in urls)
    # Order preserved (first appearance wins).
    assert "issue-42" in urls[0]
    assert "issue-41" in urls[1]
    assert "issue-40" in urls[2]


def test_extract_issue_urls_defaults_to_empty_on_empty_html():
    assert _extract_issue_urls("", "https://x.com/", "/p/") == []


def test_extract_published_from_meta_opengraph():
    html = """
    <html><head>
      <meta property="article:published_time" content="2026-04-13T12:34:56Z">
    </head><body></body></html>
    """
    out = _extract_published_from_meta(html)
    assert out is not None
    assert out.startswith("2026-04-13T12:34:56")


def test_extract_published_from_meta_time_tag_fallback():
    html = """
    <html><body>
      <article><time datetime="2026-04-10T08:00:00Z">Apr 10</time></article>
    </body></html>
    """
    assert _extract_published_from_meta(html).startswith("2026-04-10")


def test_archive_adapter_yields_raw_items(monkeypatch):
    issue_html = """
    <html><head>
      <meta property="article:published_time" content="2026-04-13T10:00:00Z">
    </head><body><article>
      <h1>MCP is eating the world</h1>
      <p>Agents, MCP, reasoning — the whole stack is shifting.</p>
    </article></body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/archive":
            return httpx.Response(200, text=ARCHIVE_HTML)
        if request.url.path.startswith("/p/"):
            return httpx.Response(200, text=issue_html)
        return httpx.Response(404)

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.sources.archive.httpx.Client", FakeClient)

    source = {
        "id": "codenewsletter",
        "type": "archive",
        "name": "Code Newsletter",
        "url": "https://codenewsletter.ai/",
        "archive_url": "https://codenewsletter.ai/archive",
        "issue_path_prefix": "/p/",
        "author": "Code Newsletter",
    }
    items = list(
        ArchiveAdapter().fetch(source, Cursor(), user_agent="t", max_items=10)
    )
    assert len(items) == 3
    assert items[0].kind == "article"
    assert items[0].source_id == "codenewsletter"
    assert items[0].author == "Code Newsletter"
    assert "MCP" in items[0].title or "eating" in items[0].title
    assert "agents" in items[0].content_text.lower()
    assert items[0].published_at.startswith("2026-04-13T10:00:00")


def test_archive_adapter_prefers_sitemap_over_html(monkeypatch):
    """When sitemap.xml has issues, it wins over HTML archive scraping."""
    sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://codenewsletter.ai/p/issue-100-latest</loc></url>
      <url><loc>https://codenewsletter.ai/p/issue-99-older</loc></url>
      <url><loc>https://codenewsletter.ai/p/issue-98-even-older</loc></url>
      <url><loc>https://codenewsletter.ai/about</loc></url>
    </urlset>
    """
    issue_html = "<html><body><article>MCP agents context window.</article></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(
                200,
                content=sitemap_xml.encode(),
                headers={"content-type": "application/xml"},
            )
        if request.url.path.startswith("/p/"):
            return httpx.Response(200, text=issue_html)
        # If we get here for /archive, the sitemap path wasn't taken.
        raise AssertionError(f"Unexpected request: {request.url.path}")

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.sources.archive.httpx.Client", FakeClient)

    source = {
        "id": "codenewsletter",
        "type": "archive",
        "url": "https://codenewsletter.ai/",
        "archive_url": "https://codenewsletter.ai/archive",
    }
    items = list(
        ArchiveAdapter().fetch(source, Cursor(), user_agent="t", max_items=10)
    )
    assert len(items) == 3  # /about is filtered by prefix
    # Order preserved from sitemap.
    assert "issue-100" in items[0].url
    assert "issue-99" in items[1].url
    assert "issue-98" in items[2].url


def test_archive_adapter_falls_back_to_html_when_sitemap_empty(monkeypatch):
    """If sitemap is empty or 404, fall back to scraping /archive HTML."""
    issue_html = "<html><body><p>some mcp body</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/sitemap.xml":
            return httpx.Response(404)
        if request.url.path == "/archive":
            return httpx.Response(200, text=ARCHIVE_HTML)
        if request.url.path.startswith("/p/"):
            return httpx.Response(200, text=issue_html)
        return httpx.Response(404)

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.sources.archive.httpx.Client", FakeClient)

    source = {
        "id": "codenewsletter",
        "type": "archive",
        "url": "https://codenewsletter.ai/",
        "archive_url": "https://codenewsletter.ai/archive",
    }
    items = list(
        ArchiveAdapter().fetch(source, Cursor(), user_agent="t", max_items=10)
    )
    assert len(items) == 3  # from ARCHIVE_HTML fallback


def test_archive_adapter_respects_cursor(monkeypatch):
    """If last_guid matches an issue canonical, stop (don't refetch history)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/archive":
            return httpx.Response(200, text=ARCHIVE_HTML)
        return httpx.Response(
            200, text="<html><body><p>some body text about mcp</p></body></html>"
        )

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.sources.archive.httpx.Client", FakeClient)

    source = {
        "id": "codenewsletter",
        "type": "archive",
        "url": "https://codenewsletter.ai/",
        "archive_url": "https://codenewsletter.ai/archive",
    }
    # Pretend we already saw Issue 41.
    cursor = Cursor(last_guid="https://codenewsletter.ai/p/issue-41-mcp-edition")
    items = list(
        ArchiveAdapter().fetch(source, cursor, user_agent="t", max_items=10)
    )
    # Only Issue 42 comes before Issue 41 in the HTML order; loop stops there.
    assert len(items) == 1
    assert "issue-42" in items[0].url_canonical
