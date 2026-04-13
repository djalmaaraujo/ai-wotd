from pathlib import Path

import httpx
import pytest

from wotd.sources.base import Cursor
from wotd.sources.rss import RssAdapter


FIXTURE = Path(__file__).parent / "fixtures" / "openai_feed.xml"


class _Handler:
    def __init__(self, feed_bytes: bytes):
        self.feed_bytes = feed_bytes

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("feed.xml") or request.url.host.endswith("example.com") and request.url.path == "/feed":
            return httpx.Response(200, content=self.feed_bytes)
        if request.url.host == "openai.example.com":
            body = b"<html><body><article><h1>Post</h1><p>MCP is everywhere.</p></article></body></html>"
            return httpx.Response(200, content=body)
        return httpx.Response(404)


def test_rss_adapter_emits_items(monkeypatch):
    feed_bytes = FIXTURE.read_bytes()
    handler = _Handler(feed_bytes)

    original_client = httpx.Client

    class FakeClient(original_client):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.sources.rss.httpx.Client", FakeClient)

    source = {
        "id": "openai-fixture",
        "type": "rss",
        "name": "OpenAI fixture",
        "feed": "https://openai.example.com/feed",
    }
    adapter = RssAdapter()
    items = list(
        adapter.fetch(source, Cursor(), user_agent="test", max_items=10)
    )
    assert len(items) == 2
    assert items[0].source_id == "openai-fixture"
    assert "MCP" in items[0].content_text or "mcp" in items[0].content_text.lower()
    assert items[0].kind == "article"
    assert items[0].url.startswith("https://openai.example.com/")
