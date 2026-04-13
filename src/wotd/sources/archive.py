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

        # 1. Load the archive index page.
        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                resp = client.get(archive_url, headers={"User-Agent": user_agent})
            resp.raise_for_status()
            archive_html = resp.text
        except Exception as exc:
            logger.warning(
                "archive: index fetch failed for %s: %s", source.get("id"), exc
            )
            return

        issue_urls = _extract_issue_urls(
            archive_html, archive_url, issue_path_prefix
        )
        if not issue_urls:
            logger.info(
                "archive: no issue URLs found on %s (prefix=%r)",
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
