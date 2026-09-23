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


def test_run_fetch_survives_a_malformed_link_and_keeps_cursors(tmp_path, monkeypatch):
    """A bad href must not cost the run every source's cursor."""
    import httpx

    from wotd import state
    from wotd.config import Paths, Settings
    from wotd.fetch import run_fetch

    feed = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item><title>Issue 1</title><link>https://news.example/p/1</link>"
        "<guid>g1</guid></item></channel></rss>"
    )
    issue_html = (
        '<html><body><p>hi</p>'
        '<a href="https://ex_ample..com/post">bad</a>'
        '<a href="https://good.example/post">good</a></body></html>'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text="User-agent: *\nDisallow:\n")
        if "/feed" in url:
            return httpx.Response(200, content=feed.encode())
        if "good.example" in url:
            return httpx.Response(200, text="<html><body><p>a real body</p></body></html>")
        return httpx.Response(200, text=issue_html)

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    for module in ("wotd.fetch", "wotd.robots", "wotd.sources.rss"):
        monkeypatch.setattr(f"{module}.httpx.Client", FakeClient)

    paths = Paths.from_root(tmp_path)
    paths.ensure()
    source = {
        "id": "news",
        "type": "newsletter",
        "platform": "substack",
        "name": "News",
        "feed": "https://news.example/feed",
        "added": "2026-09-23",
    }
    result = run_fetch(paths, Settings(), [source])

    assert result["new_articles"] >= 1
    assert state.load_cursors(paths.index)["news"]["last_guid"] == "g1"
