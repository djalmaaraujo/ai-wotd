"""Generic RSS/Atom adapter.

Uses feedparser for the feed itself, httpx for article HTML fetches, and
readability-lxml + BeautifulSoup to extract the main body.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import ClassVar, Iterable

import feedparser
import httpx
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from ..linkfollow import canonicalize
from .base import Cursor, RawItem

logger = logging.getLogger(__name__)


def _parse_date(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        dt = date_parser.parse(value)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def extract_article_text(html: str) -> tuple[str, str]:
    """Return (title, text). Uses readability when available, BS4 fallback."""
    if not html:
        return "", ""
    try:
        from readability import Document  # type: ignore

        doc = Document(html)
        title = (doc.short_title() or "").strip()
        summary_html = doc.summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        t = soup.title
        title = t.get_text(strip=True) if t else ""

    # Strip scripts/styles that readability occasionally lets through.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return title, text


def _feed_entry_text(entry) -> str:
    """Best-effort body from the RSS entry itself (works if publisher inlines)."""
    content = entry.get("content") or []
    if content:
        html = content[0].get("value") or ""
    else:
        html = entry.get("summary") or entry.get("description") or ""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n", strip=True)


class RssAdapter:
    type: ClassVar[str] = "rss"

    def fetch(
        self,
        source: dict,
        cursor: Cursor,
        *,
        user_agent: str,
        max_items: int,
        seen_urls: frozenset[str] = frozenset(),
    ) -> Iterable[RawItem]:
        feed_url = source.get("feed")
        if not feed_url:
            return
        headers = {"User-Agent": user_agent}
        if cursor.etag:
            headers["If-None-Match"] = cursor.etag
        if cursor.last_modified:
            headers["If-Modified-Since"] = cursor.last_modified

        try:
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                resp = client.get(feed_url, headers=headers)
            if resp.status_code == 304:
                logger.info("rss: %s not modified", source.get("id"))
                return
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
        except Exception as exc:
            logger.warning("rss: fetch failed for %s: %s", source.get("id"), exc)
            return

        seen = 0
        for entry in parsed.entries:
            if seen >= max_items:
                break
            guid = entry.get("id") or entry.get("guid") or entry.get("link")
            if not guid:
                continue
            if cursor.last_guid and guid == cursor.last_guid:
                break  # caught up
            url = entry.get("link") or guid
            url_canonical = canonicalize(url) or url

            # Skip entries whose canonical URL was already ingested (this
            # run or earlier). Saves a full article-HTML GET per duplicate.
            if url_canonical in seen_urls:
                continue

            published = _parse_date(
                entry.get("published") or entry.get("updated")
            )
            author = (
                entry.get("author")
                or source.get("author")
                or source.get("name")
            )
            title = (entry.get("title") or "").strip()

            # Try to fetch the full page; fall back to inline content.
            body_text = ""
            body_html: str | None = None
            try:
                with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                    article_resp = client.get(url, headers={"User-Agent": user_agent})
                if article_resp.status_code == 200:
                    ctype = (article_resp.headers.get("content-type") or "").lower()
                    # Only pass HTML/XML/text through readability. PDFs and
                    # other binary blobs get skipped — they contaminate the
                    # corpus with stream markers like "obj", "endobj", "q q".
                    if "html" in ctype or "xml" in ctype or ctype.startswith("text/"):
                        body_html = article_resp.text
                        _, body_text = extract_article_text(body_html)
                    else:
                        logger.info(
                            "rss: skipping non-HTML body for %s (content-type=%s)",
                            url,
                            ctype or "unknown",
                        )
            except Exception as exc:
                logger.info("rss: body fetch failed for %s: %s", url, exc)

            if not body_text:
                body_text = _feed_entry_text(entry)
            if not body_text and not title:
                continue

            yield RawItem(
                source_id=source["id"],
                external_id=guid,
                url=url,
                url_canonical=url_canonical,
                title=title,
                author=author,
                published_at=published,
                content_text=body_text,
                kind="article",
                content_html=body_html,
            )
            seen += 1
