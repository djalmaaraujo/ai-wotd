"""Deterministic trending-score WOTD picker."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from .terms import load_allowlist


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


def pick_wotd(
    stats_dir: Path,
    wotd_dir: Path,
    target: date,
    baseline_days: int = 30,
) -> dict | None:
    """Pick the WOTD for `target` and persist `wotd/<target>.json`.

    Returns the written payload, or None if no stats exist.
    """
    today_stats = _read_stats(stats_dir, target)
    if not today_stats:
        return None
    baseline_per_term, ever_present = _baseline(stats_dir, target, baseline_days)
    candidates = score_terms(today_stats, baseline_per_term, ever_present)

    if not candidates:
        # Still write a skeleton so the day is marked as processed.
        payload = {
            "date": target.isoformat(),
            "word": None,
            "score": 0.0,
            "candidates": [],
            "evidence_article_ids": [],
        }
    else:
        top = candidates[0]
        top10 = [c.to_dict() for c in candidates[:10]]
        payload = {
            "date": target.isoformat(),
            "word": top.term,
            "score": round(top.score, 6),
            "candidates": top10,
            "evidence_article_ids": list(top.articles)[:10],
        }

    wotd_dir.mkdir(parents=True, exist_ok=True)
    out = wotd_dir / f"{target.isoformat()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return payload
