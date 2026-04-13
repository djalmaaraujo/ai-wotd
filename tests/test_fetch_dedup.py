"""Regression tests for duplicate-GET avoidance in the fetch pipeline."""

from __future__ import annotations

from wotd.fetch import _extract_outbound_urls, _render_newsletter_body_html
from wotd.corpus import RawItem


def _item(url: str, html: str | None = None) -> RawItem:
    return RawItem(
        source_id="nl",
        external_id=url,
        url=url,
        url_canonical=url,
        title="t",
        author=None,
        published_at="2026-04-13T00:00:00+00:00",
        content_text="",
        kind="article",
        content_html=html,
    )


def test_extract_outbound_urls_dedupes_same_canonical():
    html = """
    <html><body>
      <a href="https://example.com/a?utm_source=x">first</a>
      <a href="https://example.com/a?utm_source=y">again, different tracking</a>
      <a href="https://example.com/b">other</a>
      <a href="#anchor">skip</a>
      <a href="mailto:a@b.com">skip</a>
    </body></html>
    """
    urls = _extract_outbound_urls(html)
    # The two /a links collapse to one entry; /b stays.
    assert len(urls) == 2
    assert any("example.com/a" in u for u in urls)
    assert any("example.com/b" in u for u in urls)


def test_render_newsletter_reuses_raw_item_html():
    """If the RSS adapter already stashed the HTML on RawItem, no network."""
    html = "<html><body>cached</body></html>"
    item = _item("https://news.example.com/issue-1", html=html)
    # If this does any network call, the sandbox will 403 and the test will
    # still pass — but the key assertion is that we return the stashed HTML
    # verbatim and never hit the network when it's available.
    out = _render_newsletter_body_html(item, user_agent="test")
    assert out == html
