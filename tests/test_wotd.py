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

    payload = pick_wotd(stats, wotd, today, baseline_days=7)
    assert payload is not None
    assert payload["word"] == "mcp", f"got candidates={payload['candidates']}"
    assert payload["candidates"][0]["term"] == "mcp"
    assert (wotd / f"{today.isoformat()}.json").exists()


def test_pick_wotd_returns_none_with_no_stats(tmp_path):
    assert pick_wotd(tmp_path / "stats", tmp_path / "wotd", date(2026, 4, 13)) is None


def test_pick_wotd_ignores_df_below_two(tmp_path):
    stats = tmp_path / "stats"
    wotd = tmp_path / "wotd"
    today = date(2026, 4, 13)
    # Only one doc — nothing with df >= 2 is possible.
    _write_stats(stats, today, {"mcp": (5, 1)}, doc_count=1)
    payload = pick_wotd(stats, wotd, today, baseline_days=30)
    assert payload is not None
    assert payload["word"] is None
    assert payload["candidates"] == []
