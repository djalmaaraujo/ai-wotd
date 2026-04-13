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


def cmd_process(args) -> int:
    paths = _paths_for(args)
    target = _parse_date(args.date)
    processed = state.load_processed_days(paths.index)
    result = corpus.build_day_stats(
        paths.articles, paths.stats, paths.fulltext_cache, target
    )
    if result is None:
        log.info("process: no articles for %s", target)
        return 0
    processed[target.isoformat()] = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "document_count": result["document_count"],
    }
    state.save_processed_days(paths.index, processed)
    log.info("process: %s -> %d docs", target, result["document_count"])
    return 0


def cmd_wotd(args) -> int:
    paths = _paths_for(args)
    settings = Settings.from_env()
    target = _parse_date(args.date)
    payload = wotd.pick_wotd(
        paths.stats, paths.wotd, target, baseline_days=settings.baseline_days
    )
    if payload is None:
        log.info("wotd: no stats for %s", target)
        return 0
    log.info("wotd: %s -> %s", target, payload.get("word"))
    return 0


def cmd_blurb(args) -> int:
    paths = _paths_for(args)
    settings = Settings.from_env()
    target = _parse_date(args.date)
    wotd_path = paths.wotd / f"{target.isoformat()}.json"
    if not wotd_path.exists():
        log.info("blurb: no wotd json for %s", target)
        return 0

    with open(wotd_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wotd", description="AI Word Of The Day pipeline.")
    p.add_argument("--root", default=".", help="Repo root (default: cwd).")
    p.add_argument("--sources", default="sources.yml", help="Path to sources.yml.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)

    sp = sub.add_parser("process")
    sp.add_argument("--date", default=None)
    sp.set_defaults(fn=cmd_process)

    sp = sub.add_parser("wotd")
    sp.add_argument("--date", default=None)
    sp.set_defaults(fn=cmd_wotd)

    sp = sub.add_parser("blurb")
    sp.add_argument("--date", default=None)
    sp.set_defaults(fn=cmd_blurb)

    sub.add_parser("export").set_defaults(fn=cmd_export)
    sub.add_parser("build").set_defaults(fn=cmd_build)

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
