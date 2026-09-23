from wotd.linkfollow import canonicalize, is_blocked, should_follow


def test_canonicalize_strips_tracking_and_fragment():
    url = "HTTP://Example.COM/path/?utm_source=x&id=1&fbclid=abc#frag"
    assert canonicalize(url) == "https://example.com/path?id=1"


def test_canonicalize_drops_trailing_slash_on_nonroot():
    assert canonicalize("https://example.com/a/") == "https://example.com/a"
    assert canonicalize("https://example.com/") == "https://example.com/"


def test_canonicalize_rejects_non_http():
    assert canonicalize("mailto:a@b.com") is None
    assert canonicalize("javascript:alert(1)") is None
    assert canonicalize("") is None


def test_is_blocked_social_and_extensions():
    assert is_blocked("https://x.com/foo")
    assert is_blocked("https://sub.youtube.com/watch")
    assert is_blocked("https://cdn.example.com/report.PDF")
    assert not is_blocked("https://openai.example.com/posts/mcp")


def test_should_follow_dedupes_via_cache():
    cache = {"https://example.com/a": {"article_id": "x--1"}}
    follow, canon = should_follow("https://example.com/a/?utm_source=n", cache)
    assert follow is False
    assert canon == "https://example.com/a"

    follow, canon = should_follow("https://example.com/new", cache)
    assert follow is True
    assert canon == "https://example.com/new"


def test_linkfollow_skips_what_robots_disallows(monkeypatch):
    """The README promises publishers this; it has to hold in the fetch path."""
    import httpx

    from wotd.fetch import _fetch_linked_article
    from wotd.robots import RobotsCache

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /no/\n")
        return httpx.Response(200, text="<html><body><p>body text here</p></body></html>")

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.robots.httpx.Client", FakeClient)
    monkeypatch.setattr("wotd.fetch.httpx.Client", FakeClient)

    robots = RobotsCache("ai-wotd/1.0")
    assert _fetch_linked_article("https://example.com/no/post", "ai-wotd/1.0", robots) is None
    assert _fetch_linked_article("https://example.com/yes/post", "ai-wotd/1.0", robots) is not None


def test_linkfollow_honours_the_noindex_header(monkeypatch):
    import httpx

    from wotd.fetch import _fetch_linked_article
    from wotd.robots import RobotsCache

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow:\n")
        return httpx.Response(
            200,
            text="<html><body><p>body</p></body></html>",
            headers={"X-Robots-Tag": "noindex"},
        )

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.robots.httpx.Client", FakeClient)
    monkeypatch.setattr("wotd.fetch.httpx.Client", FakeClient)

    assert (
        _fetch_linked_article("https://example.com/post", "ai-wotd/1.0", RobotsCache("ai-wotd/1.0"))
        is None
    )
