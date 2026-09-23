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


def test_run_fetch_keeps_cursors_when_a_link_cannot_be_followed(tmp_path, monkeypatch):
    """Patching httpx.Client patches the module for everyone, so the malformed
    host never reaches the IDNA encoder here. That path is covered for real in
    tests/test_robots.py; this one pins the surrounding run."""
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

    for module in ("wotd.fetch", "wotd.sources.rss"):
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


def _fake_client(monkeypatch, handler, modules=("wotd.fetch", "wotd.robots", "wotd.sources.rss")):
    import httpx

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    for module in modules:
        monkeypatch.setattr(f"{module}.httpx.Client", FakeClient)


def test_an_incomplete_source_does_not_advance_its_cursor(tmp_path, monkeypatch):
    """Aborting mid-feed must not strand the entries we never reached."""
    import httpx

    from wotd import state
    from wotd.config import Paths, Settings
    from wotd.fetch import run_fetch

    feed = (
        '<?xml version="1.0"?><rss version="2.0"><channel>'
        "<item><title>One</title><link>https://ok.example/1</link><guid>g1</guid></item>"
        "<item><title>Two</title><link>https://down.example/2</link><guid>g2</guid></item>"
        "<item><title>Three</title><link>https://ok.example/3</link><guid>g3</guid></item>"
        "</channel></rss>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://down.example"):
            return httpx.Response(503)
        if url.endswith("/robots.txt"):
            return httpx.Response(200, text="User-agent: *\nDisallow:\n")
        if "/feed" in url:
            return httpx.Response(200, content=feed.encode())
        return httpx.Response(200, text="<html><body><p>body</p></body></html>")

    _fake_client(monkeypatch, handler)
    paths = Paths.from_root(tmp_path)
    paths.ensure()
    source = {"id": "news", "type": "rss", "name": "News", "feed": "https://news.example/feed"}

    run_fetch(paths, Settings(), [source])
    assert state.load_cursors(paths.index).get("news", {}).get("last_guid") is None


def test_one_exploding_source_does_not_cost_the_others_their_cursors(tmp_path, monkeypatch):
    """The iteration is where adapters actually run, so it has to be guarded."""
    from wotd import state
    from wotd.config import Paths, Settings
    from wotd.fetch import run_fetch
    from wotd.sources import get_adapter

    class Exploding:
        type = "rss"

        def fetch(self, *a, **kw):
            yield from ()
            raise RuntimeError("boom")

    real = get_adapter

    def fake_adapter(kind):
        return Exploding() if kind == "bad" else real(kind)

    monkeypatch.setattr("wotd.fetch.get_adapter", fake_adapter)

    paths = Paths.from_root(tmp_path)
    paths.ensure()
    state.save_cursors(paths.index, {"kept": {"last_guid": "g0"}})

    result = run_fetch(paths, Settings(), [{"id": "bad", "type": "bad", "name": "Bad"}])
    assert result["new_articles"] == 0
    assert state.load_cursors(paths.index)["kept"]["last_guid"] == "g0"
