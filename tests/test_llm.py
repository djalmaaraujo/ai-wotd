import json
import os
from pathlib import Path

from wotd.llm import _parse_response, attach_blurb_to_wotd, generate_blurb


def test_parse_response_plain_json():
    text = '{"summary": "hello", "why": "because"}'
    s, w = _parse_response(text)
    assert s == "hello" and w == "because"


def test_parse_response_fenced_json():
    text = """```json
{"summary": "a", "why": "b"}
```"""
    s, w = _parse_response(text)
    assert s == "a" and w == "b"


def test_parse_response_trailing_prose():
    text = 'prefix {"summary": "s", "why": "w"} trailing'
    s, w = _parse_response(text)
    assert s == "s" and w == "w"


def test_parse_response_handles_bad_input():
    s, w = _parse_response("")
    assert s is None and w is None


def test_generate_blurb_skips_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert generate_blurb(word="mcp", candidates=[], evidence_articles=[]) is None


def test_attach_blurb_to_wotd_no_key_preserves_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    wotd_path = tmp_path / "2026-04-13.json"
    payload = {"date": "2026-04-13", "word": "mcp", "candidates": [], "evidence_article_ids": []}
    wotd_path.write_text(json.dumps(payload))
    ok = attach_blurb_to_wotd(wotd_path, evidence_articles=[])
    assert ok is False
    # File unchanged.
    assert json.loads(wotd_path.read_text()) == payload
