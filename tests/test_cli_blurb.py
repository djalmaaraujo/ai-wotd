import json
from datetime import date, datetime, timedelta, timezone

from wotd import cli


def _wotd(root, d: date, word, llm=None):
    path = root / "data" / "wotd" / f"{d.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"date": d.isoformat(), "word": word, "evidence_article_ids": []}
    if llm:
        payload["llm"] = llm
    path.write_text(json.dumps(payload))


def test_blurb_recent_fills_every_missing_summary_in_the_window(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).date()
    _wotd(tmp_path, today, "muse")
    _wotd(tmp_path, today - timedelta(days=2), "opus 5.5", llm={"summary": "done"})
    _wotd(tmp_path, today - timedelta(days=3), None)
    _wotd(tmp_path, today - timedelta(days=4), "pi 1.0")
    _wotd(tmp_path, today - timedelta(days=20), "fable 5")
    asked: list[str] = []
    monkeypatch.setattr(cli, "attach_blurb_to_wotd", lambda path, *a, **kw: asked.append(path.stem) or True)

    cli.main(["--root", str(tmp_path), "blurb", "--recent", "7"])

    assert asked == [today.isoformat(), (today - timedelta(days=4)).isoformat()]


def test_blurb_without_options_still_only_does_the_latest_word(tmp_path, monkeypatch):
    today = datetime.now(timezone.utc).date()
    _wotd(tmp_path, today - timedelta(days=1), "muse")
    _wotd(tmp_path, today - timedelta(days=2), "pi 1.0")
    asked: list[str] = []
    monkeypatch.setattr(cli, "attach_blurb_to_wotd", lambda path, *a, **kw: asked.append(path.stem) or True)

    cli.main(["--root", str(tmp_path), "blurb"])

    assert asked == [(today - timedelta(days=1)).isoformat()]
