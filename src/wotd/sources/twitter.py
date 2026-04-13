"""Twitter / X adapter.

Default delivery: Nitter RSS. Escape hatch: official X API via bearer token.
Both emit RawItem with kind="tweet" and content_text = tweet body.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from importlib import resources
from typing import ClassVar, Iterable
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from ..linkfollow import canonicalize
from .base import Cursor, RawItem

logger = logging.getLogger(__name__)


_TWEET_ID_RE = re.compile(r"/status/(\d+)")


def _default_nitter_instances() -> list[str]:
    override = os.environ.get("WOTD_NITTER_INSTANCES")
    if override:
        return [h.strip() for h in override.split(",") if h.strip()]
    path = resources.files("wotd.sources").joinpath("nitter_instances.txt")
    with path.open("r", encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def _swap_host(url: str, new_host: str) -> str:
    parts = urlparse(url)
    return parts._replace(netloc=new_host, scheme="https").geturl()


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


def _tweet_text_from_entry(entry) -> str:
    summary_html = entry.get("summary") or entry.get("description") or ""
    if not summary_html:
        return (entry.get("title") or "").strip()
    soup = BeautifulSoup(summary_html, "lxml")
    return soup.get_text(separator=" ", strip=True)


class TwitterAdapter:
    type: ClassVar[str] = "twitter"

    def fetch(
        self,
        source: dict,
        cursor: Cursor,
        *,
        user_agent: str,
        max_items: int,
    ) -> Iterable[RawItem]:
        if source.get("x_api"):
            yield from self._fetch_x_api(
                source, cursor, user_agent=user_agent, max_items=max_items
            )
        else:
            yield from self._fetch_nitter(
                source, cursor, user_agent=user_agent, max_items=max_items
            )

    # ---- Nitter path ---------------------------------------------------

    def _fetch_nitter(
        self,
        source: dict,
        cursor: Cursor,
        *,
        user_agent: str,
        max_items: int,
    ) -> Iterable[RawItem]:
        feed_url = source.get("nitter_feed")
        handle = source.get("handle")
        if not feed_url and not handle:
            return
        candidates: list[str] = []
        if feed_url:
            candidates.append(feed_url)
        if handle:
            for host in _default_nitter_instances():
                candidates.append(f"https://{host}/{handle}/rss")

        content = None
        for url in candidates:
            try:
                with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                    resp = client.get(url, headers={"User-Agent": user_agent})
                if resp.status_code == 200 and resp.content:
                    content = resp.content
                    break
            except Exception as exc:
                logger.info("twitter/nitter: %s failed: %s", url, exc)

        if content is None:
            logger.warning(
                "twitter/nitter: all instances failed for %s", source.get("id")
            )
            return

        parsed = feedparser.parse(content)
        min_likes = int(source.get("min_likes") or 0)
        exclude_replies = bool(source.get("exclude_replies"))

        seen = 0
        for entry in parsed.entries:
            if seen >= max_items:
                break
            link = entry.get("link") or ""
            m = _TWEET_ID_RE.search(link)
            tweet_id = m.group(1) if m else (entry.get("id") or link)
            if cursor.last_guid and tweet_id == cursor.last_guid:
                break

            text = _tweet_text_from_entry(entry)
            if not text:
                continue
            if exclude_replies and text.lstrip().startswith("@"):
                continue
            # Nitter RSS doesn't expose like counts reliably, so min_likes is a
            # best-effort hint we can't always enforce here. Kept for future.
            _ = min_likes

            author = source.get("name") or source.get("handle")
            # Canonical URL — prefer x.com over nitter hosts for attribution.
            handle = source.get("handle")
            canonical = (
                f"https://x.com/{handle}/status/{tweet_id}"
                if handle and tweet_id.isdigit()
                else canonicalize(link) or link
            )
            yield RawItem(
                source_id=source["id"],
                external_id=tweet_id,
                url=canonical,
                url_canonical=canonical,
                title=text[:80],
                author=author,
                published_at=_parse_iso(entry.get("published") or entry.get("updated")),
                content_text=text,
                kind="tweet",
            )
            seen += 1

    # ---- Official X API path ------------------------------------------

    def _fetch_x_api(
        self,
        source: dict,
        cursor: Cursor,
        *,
        user_agent: str,
        max_items: int,
    ) -> Iterable[RawItem]:
        token = os.environ.get("X_BEARER_TOKEN")
        handle = source.get("handle")
        if not token or not handle:
            logger.info(
                "twitter/x_api: skipped (missing X_BEARER_TOKEN or handle)"
            )
            return
        headers = {"Authorization": f"Bearer {token}", "User-Agent": user_agent}
        try:
            with httpx.Client(timeout=15.0) as client:
                user = client.get(
                    f"https://api.x.com/2/users/by/username/{handle}",
                    headers=headers,
                )
                user.raise_for_status()
                user_id = user.json()["data"]["id"]
                tweets = client.get(
                    f"https://api.x.com/2/users/{user_id}/tweets",
                    headers=headers,
                    params={
                        "max_results": min(max_items, 100),
                        "tweet.fields": "created_at,public_metrics,referenced_tweets",
                        "exclude": "replies" if source.get("exclude_replies") else "",
                    },
                )
                tweets.raise_for_status()
        except Exception as exc:
            logger.warning(
                "twitter/x_api: request failed for %s: %s", source.get("id"), exc
            )
            return

        data = tweets.json().get("data") or []
        min_likes = int(source.get("min_likes") or 0)
        for t in data[:max_items]:
            metrics = t.get("public_metrics") or {}
            if metrics.get("like_count", 0) < min_likes:
                continue
            tweet_id = t["id"]
            if cursor.last_guid and tweet_id == cursor.last_guid:
                break
            text = t.get("text") or ""
            canonical = f"https://x.com/{handle}/status/{tweet_id}"
            yield RawItem(
                source_id=source["id"],
                external_id=tweet_id,
                url=canonical,
                url_canonical=canonical,
                title=text[:80],
                author=source.get("name") or handle,
                published_at=_parse_iso(t.get("created_at")),
                content_text=text,
                kind="tweet",
            )
