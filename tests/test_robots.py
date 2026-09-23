"""Tests for robots.txt and X-Robots-Tag handling."""

from __future__ import annotations

import httpx
import pytest

from wotd.robots import RobotsCache, blocked_by_header


def _client(monkeypatch, handler) -> None:
    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.robots.httpx.Client", FakeClient)


def test_disallowed_path_is_refused(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")

    _client(monkeypatch, handler)
    robots = RobotsCache(user_agent="ai-wotd/1.0")
    assert robots.allowed("https://example.com/public/post") is True
    assert robots.allowed("https://example.com/private/post") is False


def test_robots_is_fetched_once_per_host(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text="User-agent: *\nDisallow:\n")

    _client(monkeypatch, handler)
    robots = RobotsCache(user_agent="ai-wotd/1.0")
    robots.allowed("https://example.com/a")
    robots.allowed("https://example.com/b")
    robots.allowed("https://other.com/c")
    assert len(calls) == 2


def test_a_missing_or_broken_robots_allows(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    _client(monkeypatch, handler)
    assert RobotsCache(user_agent="ai-wotd/1.0").allowed("https://example.com/a") is True


def test_a_server_error_on_robots_blocks_the_host(monkeypatch):
    """A 5xx means unknown, and unknown is not consent."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    _client(monkeypatch, handler)
    assert RobotsCache(user_agent="ai-wotd/1.0").allowed("https://example.com/a") is False


def test_noindex_header_is_honoured():
    assert blocked_by_header({"x-robots-tag": "noindex, nofollow"}) is True
    assert blocked_by_header({"X-Robots-Tag": "NOINDEX"}) is True
    assert blocked_by_header({"x-robots-tag": "all"}) is False
    assert blocked_by_header({}) is False
