import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from wotd.wotd import pick_wotd


def _write_stats(stats_dir: Path, d: date, terms: dict, doc_count: int = 3) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": d.isoformat(),
        "document_count": doc_count,
        "article_ids": [f"a{i}" for i in range(doc_count)],
        "terms": {
            term: {
                "tf": tf,
                "df": df,
                "articles": [f"a{i}" for i in range(df)],
                "authors": [],
            }
            for term, (tf, df) in terms.items()
        },
    }
    (stats_dir / f"{d.isoformat()}.json").write_text(json.dumps(payload))


def test_pick_wotd_prefers_trending_term(tmp_path):
    stats = tmp_path / "stats"
    wotd = tmp_path / "wotd"
    today = date(2026, 4, 13)
    # Baseline: "openai" appears every day, "mcp" is rare.
    for offset in range(1, 8):
        d = today - timedelta(days=offset)
        _write_stats(stats, d, {"openai": (5, 3), "mcp": (1, 2)})
    # Today: mcp spikes.
    _write_stats(stats, today, {"openai": (5, 3), "mcp": (12, 3)})

    payload = pick_wotd(stats, wotd, today, baseline_days=7, mode="off")
    assert payload is not None
    assert payload["word"] == "mcp", f"got candidates={payload['candidates']}"
    assert payload["candidates"][0]["term"] == "mcp"
    assert (wotd / f"{today.isoformat()}.json").exists()


def test_pick_wotd_returns_none_with_no_stats(tmp_path):
    assert (
        pick_wotd(tmp_path / "stats", tmp_path / "wotd", date(2026, 4, 13), mode="off")
        is None
    )


def test_pick_wotd_requires_df_two_with_multiple_docs(tmp_path):
    """With 2+ docs, terms that appear in only 1 doc are not eligible."""
    stats = tmp_path / "stats"
    wotd = tmp_path / "wotd"
    today = date(2026, 4, 13)
    _write_stats(
        stats,
        today,
        {"mcp": (5, 1), "agents": (2, 2)},
        doc_count=3,
    )
    payload = pick_wotd(stats, wotd, today, baseline_days=30, mode="off")
    assert payload is not None
    assert payload["word"] == "agents"  # df=2, eligible
    terms_seen = {c["term"] for c in payload["candidates"]}
    assert "mcp" not in terms_seen  # df=1, rejected with doc_count >= 2


def test_pick_wotd_elects_on_single_article_day(tmp_path):
    """Slow days with only 1 article should still elect something — the
    top-tf term — rather than returning word=None."""
    stats = tmp_path / "stats"
    wotd = tmp_path / "wotd"
    today = date(2026, 4, 13)
    _write_stats(
        stats,
        today,
        {"mcp": (7, 1), "reasoning": (2, 1)},
        doc_count=1,
    )
    payload = pick_wotd(stats, wotd, today, baseline_days=30, mode="off")
    assert payload is not None
    assert payload["word"] == "mcp"
    assert len(payload["candidates"]) >= 1
    assert payload["candidates"][0]["term"] == "mcp"


def _write_articles(articles_dir: Path, d: date, titles: list[str]) -> None:
    day_dir = articles_dir / d.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    for i, title in enumerate(titles):
        (day_dir / f"a{i}.json").write_text(
            json.dumps(
                {
                    "article_id": f"a{i}",
                    "source_id": "src",
                    "title": title,
                    "snippet": title,
                    "url": f"https://example.com/{i}",
                    "url_canonical": f"https://example.com/{i}",
                    "published_at": f"{d.isoformat()}T09:00:00+00:00",
                }
            )
        )


def _judge_stub(verdicts: dict, *, raises: Exception | None = None):
    from wotd.judge import Judgment

    def stub(**kwargs):
        if raises is not None:
            raise raises
        stub.calls.append(kwargs)
        return Judgment(model="jev-test", verdicts=verdicts, input_tokens=42)

    stub.calls = []
    return stub


def test_pick_wotd_elects_the_judges_survivor(tmp_path):
    from wotd.judge import Verdict

    stats, wotd_dir, articles = (tmp_path / p for p in ("stats", "wotd", "articles"))
    today = date(2026, 7, 17)
    _write_stats(stats, today, {"steps": (9, 3), "kimi k3": (4, 2)})
    _write_articles(articles, today, ["Moonshot drops Kimi K3", "Kimi K3 beats the rest"])

    judge = _judge_stub(
        {
            "steps": Verdict(0.8, 0.2, 0.3, 0.4, 1.0),
            "kimi k3": Verdict(0.93, 2.0, 0.84, 0.8, 2.37),
        }
    )
    payload = pick_wotd(
        stats, wotd_dir, today, articles_dir=articles, mode="on", judge_fn=judge
    )

    assert payload["word"] == "kimi k3"
    assert payload["judge"]["status"] == "ok"
    assert payload["judge"]["model"] == "jev-test"
    assert payload["judge"]["signals"]["kimi k3"]["specificity"] == 2.0
    assert payload["evidence_article_ids"]


def test_pick_wotd_abstains_when_nothing_survives(tmp_path):
    from wotd.judge import Verdict

    stats, wotd_dir, articles = (tmp_path / p for p in ("stats", "wotd", "articles"))
    today = date(2026, 4, 22)
    _write_stats(stats, today, {"nitter net": (8, 2)})
    _write_articles(articles, today, ["https://nitter.net/simonw", "nitter net status"])

    judge = _judge_stub({"nitter net": Verdict(0.3, 0.1, 0.1, 0.9, 0.2)})
    payload = pick_wotd(
        stats, wotd_dir, today, articles_dir=articles, mode="on", judge_fn=judge
    )

    assert payload["word"] is None
    assert payload["status"] == "quiet_day"
    assert payload["judge"]["status"] == "ok"


def test_pick_wotd_shadow_mode_keeps_the_deterministic_word(tmp_path):
    from wotd.judge import Verdict

    stats, wotd_dir, articles = (tmp_path / p for p in ("stats", "wotd", "articles"))
    today = date(2026, 7, 17)
    _write_stats(stats, today, {"steps": (9, 3), "kimi k3": (4, 2)})
    _write_articles(articles, today, ["Moonshot drops Kimi K3", "Kimi K3 beats the rest"])

    judge = _judge_stub(
        {
            "steps": Verdict(0.8, 0.2, 0.3, 0.4, 1.0),
            "kimi k3": Verdict(0.93, 2.0, 0.84, 0.8, 2.37),
        }
    )
    payload = pick_wotd(
        stats, wotd_dir, today, articles_dir=articles, mode="shadow", judge_fn=judge
    )

    assert payload["word"] == "steps"
    assert payload["judge"]["status"] == "shadow"
    assert payload["judge"]["pick"] == "kimi k3"


def test_pick_wotd_records_a_judge_failure_and_falls_back(tmp_path, caplog):
    from wotd.judge import JudgeError

    stats, wotd_dir, articles = (tmp_path / p for p in ("stats", "wotd", "articles"))
    today = date(2026, 7, 17)
    _write_stats(stats, today, {"steps": (9, 3)})
    _write_articles(articles, today, ["something"])

    judge = _judge_stub({}, raises=JudgeError("judge: TypeSafe returned 429"))
    with caplog.at_level("ERROR"):
        payload = pick_wotd(
            stats, wotd_dir, today, articles_dir=articles, mode="on", judge_fn=judge
        )

    assert payload["word"] == "steps"
    assert payload["judge"]["status"] == "error"
    assert "429" in payload["judge"]["error"]
    assert "429" in caplog.text


def test_pick_wotd_runs_without_a_judge(tmp_path):
    stats, wotd_dir = tmp_path / "stats", tmp_path / "wotd"
    today = date(2026, 7, 17)
    _write_stats(stats, today, {"steps": (9, 3)})

    payload = pick_wotd(stats, wotd_dir, today, mode="off")
    assert payload["word"] == "steps"
    assert payload["judge"]["status"] == "off"


def test_pick_wotd_skips_the_judge_when_the_day_has_no_articles(tmp_path):
    stats, wotd_dir = tmp_path / "stats", tmp_path / "wotd"
    today = date(2026, 7, 17)
    _write_stats(stats, today, {"steps": (9, 3)})

    judge = _judge_stub({})
    payload = pick_wotd(
        stats, wotd_dir, today, articles_dir=tmp_path / "articles", mode="on",
        judge_fn=judge,
    )

    assert payload["word"] == "steps"
    assert payload["judge"] == {"status": "skipped", "reason": "no_articles_on_disk"}
    assert judge.calls == []


def test_settings_read_the_judge_mode_from_the_environment(monkeypatch):
    from wotd.config import Settings

    monkeypatch.setenv("WOTD_JUDGE", " Shadow ")
    assert Settings.from_env().judge_mode == "shadow"
    monkeypatch.delenv("WOTD_JUDGE")
    assert Settings.from_env().judge_mode == "on"
