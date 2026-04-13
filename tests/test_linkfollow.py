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
