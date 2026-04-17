"""Render the static site into /docs using Jinja2."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from jinja2 import Environment, FileSystemLoader, select_autoescape


DEFAULT_SITE_BASE = "/ai-wotd"
DEFAULT_SITE_URL = "https://djalmaaraujo.github.io/ai-wotd"


def _load_day(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_days(wotd_dir: Path) -> list[dict]:
    if not wotd_dir.exists():
        return []
    out = []
    for p in sorted(wotd_dir.glob("*.json")):
        d = _load_day(p)
        if d.get("word"):
            out.append(d)
    out.sort(key=lambda d: d.get("date", ""), reverse=True)
    return out


def _articles_index(articles_dir: Path) -> dict[str, dict]:
    """Map article_id -> derivative. Used to resolve evidence refs."""
    index: dict[str, dict] = {}
    if not articles_dir.exists():
        return index
    for p in articles_dir.glob("*/*.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        index[d["article_id"]] = d
    return index


def _articles_for_date(articles_dir: Path, date_str: str) -> list[dict]:
    """Return all article derivatives for a given date, sorted by article_id."""
    day_dir = articles_dir / date_str
    if not day_dir.exists():
        return []
    result = []
    for p in sorted(day_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            result.append(json.load(f))
    return result


def _resolve_base(value: str | None) -> str:
    """Normalize a site base: strip trailing slash, keep leading slash."""
    v = value if value is not None else os.environ.get("WOTD_SITE_BASE", DEFAULT_SITE_BASE)
    v = v.rstrip("/")
    if v and not v.startswith("/"):
        v = "/" + v
    return v


def _resolve_url(value: str | None) -> str:
    v = value if value is not None else os.environ.get("WOTD_SITE_URL", DEFAULT_SITE_URL)
    return v.rstrip("/")


def render_site(
    *,
    wotd_dir: Path,
    articles_dir: Path,
    templates_dir: Path,
    docs_dir: Path,
    site_base: str | None = None,
    site_url: str | None = None,
) -> dict:
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    days = _iter_days(wotd_dir)
    articles = _articles_index(articles_dir)

    docs_dir.mkdir(parents=True, exist_ok=True)

    base = _resolve_base(site_base)
    url = _resolve_url(site_url)
    ctx = {"base": base, "site_url": url}
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # .nojekyll so GitHub Pages serves every file verbatim.
    (docs_dir / ".nojekyll").write_text("", encoding="utf-8")

    # index.html — latest day
    if days:
        latest = days[0]
        evidence = [articles[a] for a in latest.get("evidence_article_ids", []) if a in articles]
        all_day_articles = _articles_for_date(articles_dir, latest["date"])
        html = env.get_template("day.html").render(
            day=latest,
            evidence=evidence,
            all_day_articles=all_day_articles,
            is_index=True,
            all_days=days,
            now=now_iso,
            **ctx,
        )
        (docs_dir / "index.html").write_text(html, encoding="utf-8")
    else:
        (docs_dir / "index.html").write_text(
            env.get_template("empty.html").render(now=now_iso, **ctx),
            encoding="utf-8",
        )

    # Per-day permalinks.
    for i, day in enumerate(days):
        ev = [articles[a] for a in day.get("evidence_article_ids", []) if a in articles]
        all_day_arts = _articles_for_date(articles_dir, day["date"])
        prev_day = days[i + 1] if i + 1 < len(days) else None
        next_day = days[i - 1] if i > 0 else None
        html = env.get_template("day.html").render(
            day=day,
            evidence=ev,
            all_day_articles=all_day_arts,
            is_index=False,
            all_days=days,
            prev_day=prev_day,
            next_day=next_day,
            now=now_iso,
            **ctx,
        )
        day_dir = docs_dir / "d" / day["date"]
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "index.html").write_text(html, encoding="utf-8")

    # Archive.
    archive_html = env.get_template("archive.html").render(
        all_days=days, now=now_iso, **ctx
    )
    archive_dir = docs_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "index.html").write_text(archive_html, encoding="utf-8")

    # Feed.
    feed_xml = env.get_template("feed.xml").render(
        all_days=days[:50], now=now_iso, **ctx
    )
    (docs_dir / "feed.xml").write_text(feed_xml, encoding="utf-8")

    # Stylesheet (copy if not already there).
    css_src = templates_dir / "style.css"
    if css_src.exists():
        shutil.copyfile(css_src, docs_dir / "style.css")

    return {"days": len(days), "base": base, "site_url": url}
