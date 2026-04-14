"""Command-line entry point: `wotd fetch|process|wotd|blurb|export|build|run|reprocess`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import corpus, export, fetch, site, state, wotd
from .config import Paths, Settings, load_sources
from .llm import attach_blurb_to_wotd

log = logging.getLogger("wotd")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_date(value: str | None) -> date:
    if not value:
        return datetime.now(timezone.utc).date()
    return date.fromisoformat(value)


def _paths_for(args) -> Paths:
    root = Path(args.root).resolve()
    p = Paths.from_root(root)
    p.ensure()
    return p


def cmd_fetch(args) -> int:
    paths = _paths_for(args)
    settings = Settings.from_env()
    sources = load_sources(Path(args.sources))
    result = fetch.run_fetch(paths, settings, sources)
    log.info("fetch: %s", result)
    return 0


def _discover_article_days(articles_dir: Path) -> list[date]:
    """Every YYYY-MM-DD subdir that contains at least one article JSON."""
    if not articles_dir.exists():
        return []
    days: list[date] = []
    for entry in sorted(articles_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            d = date.fromisoformat(entry.name)
        except ValueError:
            continue
        if any(entry.glob("*.json")):
            days.append(d)
    return days


def _discover_stats_days(stats_dir: Path) -> list[date]:
    if not stats_dir.exists():
        return []
    days: list[date] = []
    for p in sorted(stats_dir.glob("*.json")):
        try:
            days.append(date.fromisoformat(p.stem))
        except ValueError:
            pass
    return days


def cmd_process(args) -> int:
    """Build per-day stats.

    Default: today only. With `--backfill`, rebuild every day that has
    articles — ignoring any pre-existing stats so the new day reflects
    the current state of data/articles/ (e.g. after migrate-buckets).
    """
    paths = _paths_for(args)
    processed = state.load_processed_days(paths.index)

    if args.date:
        targets = [_parse_date(args.date)]
    elif getattr(args, "backfill", False):
        targets = _discover_article_days(paths.articles)
        if not targets:
            log.info("process: no article days found")
            return 0
    else:
        targets = [datetime.now(timezone.utc).date()]

    built = 0
    for target in targets:
        result = corpus.build_day_stats(
            paths.articles, paths.stats, paths.fulltext_cache, target
        )
        if result is None:
            log.info("process: no articles for %s", target)
            continue
        processed[target.isoformat()] = {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "document_count": result["document_count"],
        }
        log.info("process: %s -> %d docs", target, result["document_count"])
        built += 1

    state.save_processed_days(paths.index, processed)
    log.info("process: built %d day(s)", built)
    return 0


def cmd_wotd(args) -> int:
    """Elect the word of the day.

    Default: today only. With `--backfill`, re-elect for every stats
    day — overwriting prior wotd/*.json so stale elections (from before
    a stats rebuild) don't linger.
    """
    paths = _paths_for(args)
    settings = Settings.from_env()

    if args.date:
        targets = [_parse_date(args.date)]
    elif getattr(args, "backfill", False):
        targets = _discover_stats_days(paths.stats)
        if not targets:
            log.info("wotd: no stats found")
            return 0
    else:
        targets = [datetime.now(timezone.utc).date()]

    for target in targets:
        payload = wotd.pick_wotd(
            paths.stats, paths.wotd, target, baseline_days=settings.baseline_days
        )
        if payload is None:
            log.info("wotd: no stats for %s", target)
            continue
        log.info("wotd: %s -> %s", target, payload.get("word"))
    return 0


def _latest_wotd_date(wotd_dir: Path) -> date | None:
    days: list[date] = []
    for p in wotd_dir.glob("*.json"):
        try:
            days.append(date.fromisoformat(p.stem))
        except ValueError:
            pass
    return max(days) if days else None


def cmd_blurb(args) -> int:
    paths = _paths_for(args)
    settings = Settings.from_env()
    if args.date:
        target = _parse_date(args.date)
    else:
        # Default: most recent day that actually has a word elected.
        latest = None
        for p in sorted(paths.wotd.glob("*.json"), reverse=True):
            try:
                payload = json.loads(p.read_text())
            except Exception:
                continue
            if payload.get("word"):
                latest = date.fromisoformat(p.stem)
                break
        if latest is None:
            log.info("blurb: no elected word to blurb yet")
            return 0
        target = latest

    wotd_path = paths.wotd / f"{target.isoformat()}.json"
    if not wotd_path.exists():
        log.info("blurb: no wotd json for %s", target)
        return 0

    with open(wotd_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Don't re-blurb if it already has one (avoid burning API quota daily).
    if payload.get("llm", {}).get("summary") and not getattr(args, "force", False):
        log.info("blurb: %s already has a blurb; skipping", target)
        return 0

    # Collect evidence with full text (from cache if present).
    evidence_articles: list[dict] = []
    for article_id in payload.get("evidence_article_ids", [])[:8]:
        # find the derivative
        for day_dir in paths.articles.iterdir():
            candidate = day_dir / f"{article_id}.json"
            if candidate.exists():
                with open(candidate, "r", encoding="utf-8") as f:
                    d = json.load(f)
                full = corpus.load_full_text(article_id, paths.fulltext_cache) or ""
                d["content_text"] = full or d.get("snippet") or ""
                evidence_articles.append(d)
                break

    ok = attach_blurb_to_wotd(
        wotd_path,
        evidence_articles,
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
    )
    log.info("blurb: %s -> %s", target, "written" if ok else "skipped")
    return 0


def cmd_export(args) -> int:
    paths = _paths_for(args)
    result = export.export_all(paths.articles, paths.stats, paths.wotd, paths.parquet)
    log.info("export: %s", result)
    return 0


def cmd_build(args) -> int:
    paths = _paths_for(args)
    templates_dir = paths.root / "templates"
    result = site.render_site(
        wotd_dir=paths.wotd,
        articles_dir=paths.articles,
        templates_dir=templates_dir,
        docs_dir=paths.docs,
    )
    log.info("build: %s", result)
    return 0


def cmd_run(args) -> int:
    """Full pipeline: fetch → process → wotd → blurb → export → build."""
    rc = cmd_fetch(args)
    if rc:
        return rc
    rc = cmd_process(args)
    if rc:
        return rc
    rc = cmd_wotd(args)
    if rc:
        return rc
    rc = cmd_blurb(args)
    if rc:
        return rc
    rc = cmd_export(args)
    if rc:
        return rc
    return cmd_build(args)


def cmd_reprocess(args) -> int:
    paths = _paths_for(args)
    settings = Settings.from_env()
    frm = _parse_date(args.frm)
    to = _parse_date(args.to)
    if to < frm:
        log.error("reprocess: --to %s is before --from %s", to, frm)
        return 2
    steps = args.steps
    current = frm
    processed = state.load_processed_days(paths.index)
    while current <= to:
        key = current.isoformat()
        if steps in ("process", "process+wotd", "all"):
            # Remove prior stats so the rebuild is clean.
            stats_path = paths.stats / f"{key}.json"
            if stats_path.exists():
                stats_path.unlink()
            processed.pop(key, None)
            corpus.build_day_stats(
                paths.articles, paths.stats, paths.fulltext_cache, current
            )
            processed[key] = {
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "reprocessed": True,
            }
        if steps in ("wotd", "process+wotd", "all"):
            wp = paths.wotd / f"{key}.json"
            if wp.exists():
                wp.unlink()
            wotd.pick_wotd(
                paths.stats, paths.wotd, current, baseline_days=settings.baseline_days
            )
        current = current + timedelta(days=1)
    state.save_processed_days(paths.index, processed)
    export.export_all(paths.articles, paths.stats, paths.wotd, paths.parquet)
    return 0


def cmd_migrate_buckets(args) -> int:
    """Re-bucket existing article JSONs to match the current policy.

    Policy: bucket by `published_at` when plausible (between
    2015-01-01 and today inclusive), else fall back to `fetched_at`.
    Idempotent; safe to run repeatedly. Cleans up day folders that end
    up empty.
    """
    paths = _paths_for(args)
    articles_dir = paths.articles
    if not articles_dir.exists():
        log.info("migrate: no data/articles/ yet")
        return 0

    today = datetime.now(timezone.utc).date()

    def _parse(iso: str | None) -> date | None:
        if not iso:
            return None
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00")).date()
        except (ValueError, AttributeError):
            return None

    moved = 0
    skipped = 0
    for json_path in sorted(articles_dir.glob("*/*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("migrate: unreadable %s: %s", json_path, exc)
            continue
        pub = _parse(payload.get("published_at"))
        fetched = _parse(payload.get("fetched_at"))
        earliest = date(2015, 1, 1)
        target: date | None = None
        if pub and earliest <= pub <= today:
            target = pub
        elif fetched:
            target = fetched
        if target is None:
            log.info(
                "migrate: skipping %s (no usable published_at/fetched_at)",
                json_path.name,
            )
            skipped += 1
            continue
        bucket = target.isoformat()
        if json_path.parent.name == bucket:
            continue  # already in the right place
        new_dir = articles_dir / bucket
        new_dir.mkdir(parents=True, exist_ok=True)
        new_path = new_dir / json_path.name
        if new_path.exists():
            # Collision — keep whichever has content_sha256 that matches;
            # otherwise leave the source alone.
            log.info(
                "migrate: %s already exists at %s; leaving original",
                json_path.name,
                bucket,
            )
            continue
        json_path.rename(new_path)
        moved += 1

    # Clean up empty day folders.
    for day_dir in list(articles_dir.iterdir()):
        if day_dir.is_dir() and not any(day_dir.iterdir()):
            day_dir.rmdir()

    log.info("migrate: moved %d file(s), skipped %d", moved, skipped)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wotd", description="AI Word Of The Day pipeline.")
    p.add_argument("--root", default=".", help="Repo root (default: cwd).")
    p.add_argument("--sources", default="sources.yml", help="Path to sources.yml.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)

    sp = sub.add_parser("process")
    sp.add_argument("--date", default=None)
    sp.add_argument(
        "--backfill",
        action="store_true",
        help="Process every article day that lacks stats (opt-in, slow).",
    )
    sp.set_defaults(fn=cmd_process)

    sp = sub.add_parser("wotd")
    sp.add_argument("--date", default=None)
    sp.add_argument(
        "--backfill",
        action="store_true",
        help="Elect words for every stats day without a wotd yet.",
    )
    sp.set_defaults(fn=cmd_wotd)

    sp = sub.add_parser("blurb")
    sp.add_argument("--date", default=None)
    sp.add_argument("--force", action="store_true", help="Re-blurb even if already present.")
    sp.set_defaults(fn=cmd_blurb)

    sub.add_parser("export").set_defaults(fn=cmd_export)
    sub.add_parser("build").set_defaults(fn=cmd_build)
    sub.add_parser("migrate-buckets").set_defaults(fn=cmd_migrate_buckets)

    sp = sub.add_parser("run")
    sp.add_argument("--date", default=None)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("reprocess")
    sp.add_argument("--from", dest="frm", required=True)
    sp.add_argument("--to", dest="to", required=True)
    sp.add_argument(
        "--steps",
        choices=["process", "wotd", "process+wotd", "all"],
        default="process+wotd",
    )
    sp.set_defaults(fn=cmd_reprocess)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
