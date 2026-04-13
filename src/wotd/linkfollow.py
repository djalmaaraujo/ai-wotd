"""One-hop link following: canonicalize URLs, apply blocklist, dedupe via url_cache."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_STRIP_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "utm_name",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "s",
    "t",
    "igshid",
    "spm",
}

_BLOCKED_EXTENSIONS = (
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".mp3",
    ".mp4",
    ".mov",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".exe",
    ".dmg",
    ".pkg",
)


def load_blocklist() -> frozenset[str]:
    path = resources.files("wotd.resources").joinpath("linkfollow_blocklist.txt")
    with path.open("r", encoding="utf-8") as f:
        return frozenset(
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        )


def canonicalize(url: str) -> str | None:
    """Return a canonical URL, or None if the input isn't a usable http(s) URL."""
    if not url:
        return None
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    if not parts.hostname:
        return None

    # Lowercase host, drop default ports, strip fragment.
    host = parts.hostname.lower()
    netloc = host
    if parts.port and not (
        (parts.scheme == "http" and parts.port == 80)
        or (parts.scheme == "https" and parts.port == 443)
    ):
        netloc = f"{host}:{parts.port}"

    # Drop tracking params, keep rest in original order.
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in _STRIP_PARAMS
    ]
    query = urlencode(kept, doseq=True)

    # Normalize path: remove trailing slash on non-root paths.
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    return urlunsplit(("https", netloc, path, query, ""))


def is_blocked(url: str, blocklist: frozenset[str] | None = None) -> bool:
    """True if the URL points at a blocklisted domain or file extension."""
    blocklist = blocklist if blocklist is not None else load_blocklist()
    try:
        parts = urlsplit(url)
    except ValueError:
        return True
    host = (parts.hostname or "").lower()
    if not host:
        return True

    # Direct match or subdomain of any blocked host.
    for blocked in blocklist:
        if host == blocked or host.endswith("." + blocked):
            return True

    # File extensions.
    path = (parts.path or "").lower()
    if path.endswith(_BLOCKED_EXTENSIONS):
        return True

    return False


def should_follow(
    url: str,
    url_cache: dict,
    blocklist: frozenset[str] | None = None,
) -> tuple[bool, str | None]:
    """Decide if a URL should be fetched. Returns (follow, canonical_url)."""
    canonical = canonicalize(url)
    if canonical is None:
        return False, None
    if is_blocked(canonical, blocklist):
        return False, canonical
    if canonical in url_cache:
        return False, canonical
    return True, canonical
