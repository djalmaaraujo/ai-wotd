"""Trending-score WOTD picker, gated by the TypeSafe judge."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from .judge import JudgeError, judge_candidates, rank
from .terms import build_candidate_pool, load_allowlist

logger = logging.getLogger(__name__)

BODY_CANDIDATES = 40
RECENT_DAYS = 7
MAX_RECENT_PER_DAY = 12
MAX_TITLE_CHARS = 110
MAX_SIGNALS = 15


@dataclass
class Candidate:
    term: str
    score: float
    tf_today: int
    df_today: int
    avg_tf_baseline: float
    articles: list[str]

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "score": round(self.score, 6),
            "tf_today": self.tf_today,
            "df_today": self.df_today,
            "avg_tf_baseline": round(self.avg_tf_baseline, 4),
            "articles": list(self.articles),
        }


def _read_stats(stats_dir: Path, d: date) -> dict | None:
    path = stats_dir / f"{d.isoformat()}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _baseline(
    stats_dir: Path, target: date, baseline_days: int
) -> tuple[dict[str, list[int]], set[str]]:
    """Collect per-term tf lists over the last `baseline_days` days
    strictly before `target`, plus the set of terms that appeared every day
    (for long-running-topic demotion)."""
    per_term: dict[str, list[int]] = {}
    daily_term_sets: list[set[str]] = []
    for offset in range(1, baseline_days + 1):
        d = target - timedelta(days=offset)
        data = _read_stats(stats_dir, d)
        if not data:
            continue
        terms = data.get("terms", {})
        daily_term_sets.append(set(terms.keys()))
        for term, info in terms.items():
            per_term.setdefault(term, []).append(int(info.get("tf", 0)))

    ever_present: set[str] = set()
    if daily_term_sets:
        ever_present = set.intersection(*daily_term_sets)

    return per_term, ever_present


def score_terms(
    today_stats: dict,
    baseline_per_term: dict[str, list[int]],
    ever_present: set[str],
    allowlist: frozenset[str] | None = None,
) -> list[Candidate]:
    allowlist = allowlist if allowlist is not None else load_allowlist()
    today_terms: dict = today_stats.get("terms", {})
    today_doc_count = int(today_stats.get("document_count", 1) or 1)

    # When the day only has one article, df>=2 is impossible — don't filter
    # on it; otherwise there is no word of the day on slow days.
    min_df = 2 if today_doc_count >= 2 else 1

    candidates: list[Candidate] = []
    for term, info in today_terms.items():
        tf_today = int(info.get("tf", 0))
        df_today = int(info.get("df", 0))
        if df_today < min_df:
            continue

        baseline_tfs = baseline_per_term.get(term, [])
        avg_tf_baseline = (
            sum(baseline_tfs) / max(len(baseline_tfs), 1) if baseline_tfs else 0.0
        )

        trend = tf_today / max(avg_tf_baseline, 0.5)
        df_weight = df_today / today_doc_count  # fraction of today's docs
        score = math.log(1.0 + tf_today) * trend * (0.5 + df_weight)

        if term in allowlist:
            score *= 1.5
        if term in ever_present:
            score *= 0.5

        candidates.append(
            Candidate(
                term=term,
                score=score,
                tf_today=tf_today,
                df_today=df_today,
                avg_tf_baseline=avg_tf_baseline,
                articles=list(info.get("articles", [])),
            )
        )

    candidates.sort(key=lambda c: (-c.score, c.term))
    return candidates


def _day_articles(articles_dir: Path | None, target: date) -> list[dict]:
    if articles_dir is None:
        return []
    day_dir = articles_dir / target.isoformat()
    if not day_dir.exists():
        return []
    out = []
    for path in sorted(day_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _recent_titles(
    articles_dir: Path | None, target: date, days: int = RECENT_DAYS
) -> list[str]:
    titles: list[str] = []
    for offset in range(1, days + 1):
        for article in _day_articles(articles_dir, target - timedelta(days=offset))[
            :MAX_RECENT_PER_DAY
        ]:
            title = article.get("title")
            if title:
                titles.append(title[:MAX_TITLE_CHARS])
    return titles


def _evidence_for(term: str, today_stats: dict, articles: list[dict]) -> list[str]:
    """Article ids that back a word, whether or not it came from the body stats."""
    from_stats = today_stats.get("terms", {}).get(term, {}).get("articles", [])
    if from_stats:
        return list(dict.fromkeys(from_stats))[:10]
    needle = term.lower()
    matched = [
        a["article_id"]
        for a in articles
        if needle in ((a.get("title") or "") + " " + (a.get("snippet") or "")).lower()
    ]
    return matched[:10]


def _judge_payload(
    candidates: list[Candidate],
    today_stats: dict,
    articles: list[dict],
    recent: list[str],
    target: date,
    judge_fn,
) -> tuple[dict, list[str]]:
    """Run the judge over the day's candidate pool. Never raises."""
    pool = build_candidate_pool(
        [c.term for c in candidates[:BODY_CANDIDATES]],
        [a.get("title") for a in articles],
    )
    try:
        judgment = judge_fn(
            terms=pool,
            articles=articles,
            recent_titles=recent,
            date=target.isoformat(),
        )
    except JudgeError as exc:
        logger.error("wotd: the judge failed, falling back to the scorer: %s", exc)
        return {"status": "error", "error": str(exc)}, []

    survivors = rank(judgment.verdicts)
    meta = {
        "status": "ok",
        "model": judgment.model,
        "input_tokens": judgment.input_tokens,
        "pool_size": len(pool),
        "survivors": survivors[:10],
        "signals": {
            term: judgment.verdicts[term].to_dict() for term in survivors[:MAX_SIGNALS]
        },
    }
    return meta, survivors


def pick_wotd(
    stats_dir: Path,
    wotd_dir: Path,
    target: date,
    baseline_days: int = 30,
    *,
    articles_dir: Path | None = None,
    mode: str | None = None,
    judge_fn=judge_candidates,
) -> dict | None:
    """Pick the WOTD for `target` and persist `wotd/<target>.json`.

    The scorer nominates, the judge filters. In `shadow` mode the judge runs and
    is recorded but the scorer's pick still ships. Returns the written payload,
    or None if no stats exist.
    """
    mode = (mode or os.environ.get("WOTD_JUDGE") or "on").lower()
    today_stats = _read_stats(stats_dir, target)
    if not today_stats:
        return None
    baseline_per_term, ever_present = _baseline(stats_dir, target, baseline_days)
    candidates = score_terms(today_stats, baseline_per_term, ever_present)

    payload = {
        "date": target.isoformat(),
        "word": None,
        "score": 0.0,
        "candidates": [c.to_dict() for c in candidates[:10]],
        "evidence_article_ids": [],
        "judge": {"status": "off"},
    }

    if candidates:
        det_top = candidates[0]
        chosen_term = det_top.term
        articles: list[dict] = []

        articles = _day_articles(articles_dir, target)
        if mode in ("on", "shadow") and not articles:
            payload["judge"] = {"status": "skipped", "reason": "no_articles_on_disk"}
            logger.warning(
                "wotd: %s has no article derivatives; the judge cannot read the day",
                target,
            )
        elif mode in ("on", "shadow"):
            recent = _recent_titles(articles_dir, target)
            meta, survivors = _judge_payload(
                candidates, today_stats, articles, recent, target, judge_fn
            )
            judged_term = survivors[0] if survivors else None
            if mode == "shadow":
                meta["status"] = "shadow" if meta["status"] == "ok" else meta["status"]
                meta["pick"] = judged_term
            elif meta["status"] == "ok":
                chosen_term = judged_term
            payload["judge"] = meta

        if chosen_term is None:
            payload["status"] = "quiet_day"
        else:
            scored = next((c for c in candidates if c.term == chosen_term), None)
            payload["word"] = chosen_term
            payload["score"] = round(scored.score, 6) if scored else 0.0
            payload["evidence_article_ids"] = _evidence_for(
                chosen_term, today_stats, articles
            )

    wotd_dir.mkdir(parents=True, exist_ok=True)
    out = wotd_dir / f"{target.isoformat()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return payload
