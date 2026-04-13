"""Fetch pipeline: dispatch to adapters, follow newsletter links one hop,
write per-article derivatives, maintain url_cache and cursors.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

from . import state
from .config import Paths, Settings
from .corpus import RawItem, article_id_for, write_article_derivative
from .linkfollow import canonicalize, is_blocked, load_blocklist
from .sources import get_adapter
from .sources.base import Cursor
from .sources.rss import extract_article_text

logger = logging.getLogger(__name__)


def _render_newsletter_body_html(item: RawItem, user_agent: str) -> str:
    """Return the HTML of the newsletter issue.

    Reuses the HTML the RSS adapter already fetched (on RawItem.content_html).
    Only refetches as a last resort if the adapter didn't keep it.
    """
    if item.content_html:
        return item.content_html
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(item.url, headers={"User-Agent": user_agent})
        if resp.status_code == 200:
            return resp.text
    except Exception as exc:
        logger.info("linkfollow: body refetch failed for %s: %s", item.url, exc)
    return ""


def _extract_outbound_urls(html: str, origin_url: str | None = None) -> list[str]:
    """Extract outbound hrefs from HTML, deduped by their canonicalized form.

    Order of first appearance is preserved so the most-prominently-linked
    article wins the linkfollow budget.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:"):
            continue
        canonical = canonicalize(href) or href
        if canonical in seen:
            continue
        seen.add(canonical)
        urls.append(href)
    return urls


def _fetch_linked_article(url: str, user_agent: str) -> tuple[str, str] | None:
    """Return (title, text) or None."""
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": user_agent})
        if resp.status_code != 200 or not resp.text:
            return None
        title, text = extract_article_text(resp.text)
        if not text:
            return None
        return title, text
    except Exception as exc:
        logger.info("linkfollow: fetch failed for %s: %s", url, exc)
        return None


def _follow_links_from_newsletter(
    item: RawItem,
    *,
    settings: Settings,
    blocklist: frozenset[str],
    url_cache: dict,
) -> Iterable[RawItem]:
    html = _render_newsletter_body_html(item, settings.user_agent)
    urls = _extract_outbound_urls(html, origin_url=item.url)
    followed = 0
    for url in urls:
        if followed >= settings.linkfollow_max_per_issue:
            break
        canonical = canonicalize(url)
        if canonical is None:
            continue
        if is_blocked(canonical, blocklist):
            continue
        if canonical in url_cache:
            continue
        # Skip self-references (same host).
        if item.url_canonical and canonical.split("/")[2] == item.url_canonical.split("/")[2]:
            continue

        fetched = _fetch_linked_article(canonical, settings.user_agent)
        if fetched is None:
            # Mark as seen anyway so we don't retry forever.
            url_cache[canonical] = {
                "article_id": None,
                "first_seen_date": datetime.now(timezone.utc).date().isoformat(),
                "status": "fetch_failed",
            }
            continue
        title, text = fetched
        article = RawItem(
            source_id=item.source_id,
            external_id=canonical,
            url=canonical,
            url_canonical=canonical,
            title=title or url,
            author=None,
            published_at=item.published_at,
            content_text=text,
            kind="article",
            via_source_id=item.source_id,
        )
        yield article
        followed += 1


def _upsert_url_cache(
    url_cache: dict, canonical: str, article_id: str, d: str
) -> bool:
    """Insert if absent; return True if it was a new entry."""
    if canonical in url_cache and url_cache[canonical].get("article_id"):
        return False
    url_cache[canonical] = {
        "article_id": article_id,
        "first_seen_date": d,
    }
    return True


def run_fetch(paths: Paths, settings: Settings, sources: list[dict]) -> dict:
    paths.ensure()
    cursors = state.load_cursors(paths.index)
    url_cache = state.load_url_cache(paths.index)
    blocklist = load_blocklist()
    today_iso = datetime.now(timezone.utc).date().isoformat()

    new_articles = 0
    per_source_counts: dict[str, int] = {}

    for source in sources:
        sid = source["id"]
        cursor = Cursor.from_dict(cursors.get(sid))
        try:
            adapter = get_adapter(source["type"])
        except ValueError as exc:
            logger.error("fetch: %s", exc)
            continue

        newest_guid: str | None = None
        items_seen = 0
        # Snapshot the URLs already ingested so the adapter can skip the
        # article-body GET for duplicates before making the HTTP request.
        seen_urls = frozenset(
            u for u, v in url_cache.items() if v.get("article_id")
        )
        try:
            raw_items = adapter.fetch(
                source,
                cursor,
                user_agent=settings.user_agent,
                max_items=settings.max_articles_per_source,
                seen_urls=seen_urls,
            )
        except Exception as exc:
            logger.warning("fetch: adapter crashed for %s: %s", sid, exc)
            continue

        for item in raw_items:
            if newest_guid is None:
                newest_guid = item.external_id
            # Canonicalize URL even if the adapter forgot.
            item.url_canonical = item.url_canonical or (canonicalize(item.url) or item.url)

            if item.url_canonical in url_cache and url_cache[item.url_canonical].get(
                "article_id"
            ):
                continue

            # Write derivative.
            _, derivative = write_article_derivative(
                item, paths.articles, paths.fulltext_cache
            )
            _upsert_url_cache(
                url_cache,
                item.url_canonical,
                derivative["article_id"],
                today_iso,
            )
            new_articles += 1
            items_seen += 1

            # Link-follow for newsletters.
            if source["type"] == "newsletter":
                for linked in _follow_links_from_newsletter(
                    item,
                    settings=settings,
                    blocklist=blocklist,
                    url_cache=url_cache,
                ):
                    _, linked_derivative = write_article_derivative(
                        linked, paths.articles, paths.fulltext_cache
                    )
                    _upsert_url_cache(
                        url_cache,
                        linked.url_canonical,
                        linked_derivative["article_id"],
                        today_iso,
                    )
                    new_articles += 1
                    items_seen += 1

        per_source_counts[sid] = items_seen
        if newest_guid:
            cursor.last_guid = newest_guid
            cursor.last_fetched_at = datetime.now(timezone.utc).isoformat()
            cursors[sid] = cursor.to_dict()

    state.save_cursors(paths.index, cursors)
    state.save_url_cache(paths.index, url_cache)

    return {"new_articles": new_articles, "per_source": per_source_counts}
