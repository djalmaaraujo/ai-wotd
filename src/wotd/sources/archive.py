"""Archive-page adapter.

Scrapes a "list of issues" HTML page (Beehiiv / Substack / Ghost
`/archive`) for outbound links matching a path prefix, then fetches
each issue URL and extracts the body with readability.

Use this when a publisher doesn't expose RSS at all or the RSS feed
is broken/disabled. Always try `type: rss` with `feed: .../feed` or
`feed: .../rss` FIRST — it's one roundtrip vs one-per-issue.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import ClassVar, Iterable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from ..linkfollow import canonicalize
from .base import Cursor, RawItem
from .rss import extract_article_text

logger = logging.getLogger(__name__)


DEFAULT_ISSUE_PATH_PREFIX = "/p/"  # Beehiiv / Substack convention
SITEMAP_MAX_DEPTH = 3


def _fetch_sitemap_urls(
    sitemap_url: str,
    issue_path_prefix: str,
    *,
    user_agent: str,
    _depth: int = 0,
) -> list[str]:
    """Recursively collect issue URLs from a sitemap.xml.

    Handles both `<urlset>` (leaf) and `<sitemapindex>` (parent) shapes.
    Dedupe is caller's job; this preserves sitemap order (usually
    newest-first for Beehiiv/Substack/Ghost).
    """
    if _depth >= SITEMAP_MAX_DEPTH:
        return []
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(sitemap_url, headers={"User-Agent": user_agent})
    except Exception as exc:
        logger.info("archive: sitemap fetch failed for %s: %s", sitemap_url, exc)
        return []
    if resp.status_code != 200 or not resp.content:
        return []

    try:
        soup = BeautifulSoup(resp.content, "lxml-xml")
    except Exception:
        soup = BeautifulSoup(resp.text, "html.parser")

    urls: list[str] = []
    # <sitemapindex> → recurse into child sitemaps.
    for sm in soup.find_all("sitemap"):
        loc = sm.find("loc")
        if loc and loc.text:
            urls.extend(
                _fetch_sitemap_urls(
                    loc.text.strip(),
                    issue_path_prefix,
                    user_agent=user_agent,
                    _depth=_depth + 1,
                )
            )

    # <urlset> → pull issue URLs, filter by path prefix.
    for u in soup.find_all("url"):
        loc = u.find("loc")
        if loc and loc.text:
            href = loc.text.strip()
            path = urlparse(href).path
            if path.startswith(issue_path_prefix):
                urls.append(href)

    return urls


def _parse_iso(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        dt = date_parser.parse(value)
    except Exception:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _extract_published_from_meta(html: str) -> str | None:
    """Pull a publish timestamp from OpenGraph / article meta tags."""
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    for prop in (
        "article:published_time",
        "og:article:published_time",
        "article:published",
    ):
        el = soup.find("meta", attrs={"property": prop}) or soup.find(
            "meta", attrs={"name": prop}
        )
        if el and el.get("content"):
            return _parse_iso(el["content"])
    # <time datetime="...">
    t = soup.find("time", attrs={"datetime": True})
    if t and t.get("datetime"):
        return _parse_iso(t["datetime"])
    return None


def _extract_issue_urls(
    html: str,
    base_url: str,
    issue_path_prefix: str,
) -> list[str]:
    """Return unique issue URLs on the archive page, in document order."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    base_host = (urlparse(base_url).hostname or "").lower()

    urls: list[str] = []
    seen_canonical: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue
        absolute = urljoin(base_url, href)
        parts = urlparse(absolute)
        host = (parts.hostname or "").lower()
        # Same host only (avoid outbound references).
        if host and base_host and host != base_host:
            continue
        if not parts.path.startswith(issue_path_prefix):
            continue
        canonical = canonicalize(absolute) or absolute
        if canonical in seen_canonical:
            continue
        seen_canonical.add(canonical)
        urls.append(absolute)
    return urls


class ArchiveAdapter:
    type: ClassVar[str] = "archive"

    def fetch(
        self,
        source: dict,
        cursor: Cursor,
        *,
        user_agent: str,
        max_items: int,
        seen_urls: frozenset[str] = frozenset(),
    ) -> Iterable[RawItem]:
        archive_url = source.get("archive_url") or source.get("url")
        if not archive_url:
            return
        issue_path_prefix = source.get("issue_path_prefix") or DEFAULT_ISSUE_PATH_PREFIX

        # 1. Prefer sitemap.xml — one roundtrip, full archive (no JS
        # pagination limits). Explicit sitemap_url overrides the default
        # guess at <host>/sitemap.xml.
        sitemap_url = source.get("sitemap_url")
        if not sitemap_url:
            parts = urlparse(archive_url)
            if parts.scheme and parts.hostname:
                sitemap_url = f"{parts.scheme}://{parts.hostname}/sitemap.xml"

        issue_urls: list[str] = []
        if sitemap_url:
            sitemap_urls = _fetch_sitemap_urls(
                sitemap_url, issue_path_prefix, user_agent=user_agent
            )
            # Dedupe preserving order.
            seen_canon: set[str] = set()
            for href in sitemap_urls:
                canon = canonicalize(href) or href
                if canon in seen_canon:
                    continue
                seen_canon.add(canon)
                issue_urls.append(href)
            if issue_urls:
                logger.info(
                    "archive: %d issue URLs from sitemap %s",
                    len(issue_urls),
                    sitemap_url,
                )

        # 2. Fallback to HTML scraping of the archive page.
        if not issue_urls:
            try:
                with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                    resp = client.get(
                        archive_url, headers={"User-Agent": user_agent}
                    )
                resp.raise_for_status()
                archive_html = resp.text
            except Exception as exc:
                logger.warning(
                    "archive: index fetch failed for %s: %s",
                    source.get("id"),
                    exc,
                )
                return

            issue_urls = _extract_issue_urls(
                archive_html, archive_url, issue_path_prefix
            )

        if not issue_urls:
            logger.info(
                "archive: no issue URLs found (sitemap + HTML) for %s (prefix=%r)",
                archive_url,
                issue_path_prefix,
            )
            return

        # 2. Visit each issue, extract body.
        yielded = 0
        for url in issue_urls:
            if yielded >= max_items:
                break

            canonical = canonicalize(url) or url
            if canonical in seen_urls:
                continue
            if cursor.last_guid and canonical == cursor.last_guid:
                break  # caught up with last run's most-recent

            try:
                with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                    issue_resp = client.get(url, headers={"User-Agent": user_agent})
            except Exception as exc:
                logger.info("archive: issue fetch failed for %s: %s", url, exc)
                continue
            if issue_resp.status_code != 200:
                continue
            ctype = (issue_resp.headers.get("content-type") or "").lower()
            if not ("html" in ctype or "xml" in ctype or ctype.startswith("text/")):
                logger.info(
                    "archive: skipping non-HTML issue %s (content-type=%s)",
                    url,
                    ctype or "unknown",
                )
                continue

            issue_html = issue_resp.text
            title, body_text = extract_article_text(issue_html)
            if not body_text and not title:
                continue

            published = (
                _extract_published_from_meta(issue_html)
                or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            )

            yield RawItem(
                source_id=source["id"],
                external_id=canonical,
                url=url,
                url_canonical=canonical,
                title=(title or url).strip(),
                author=source.get("author"),
                published_at=published,
                content_text=body_text,
                kind="article",
                content_html=issue_html,
            )
            yielded += 1
