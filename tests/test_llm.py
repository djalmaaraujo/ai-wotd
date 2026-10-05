import json
import os
from pathlib import Path

import httpx

from wotd.llm import _parse_response, attach_blurb_to_wotd, generate_blurb


def test_parse_response_plain_json():
    text = '{"summary": "hello", "why": "because", "definition": {"text": "a thing", "references": ["https://example.com"]}}'
    s, w, d = _parse_response(text)
    assert s == "hello" and w == "because"
    assert d == {"text": "a thing", "references": ["https://example.com"]}


def test_parse_response_fenced_json():
    text = """```json
{"summary": "a", "why": "b", "definition": {"text": "def text", "references": []}}
```"""
    s, w, d = _parse_response(text)
    assert s == "a" and w == "b"
    assert d == {"text": "def text", "references": []}


def test_parse_response_trailing_prose():
    text = 'prefix {"summary": "s", "why": "w"} trailing'
    s, w, d = _parse_response(text)
    assert s == "s" and w == "w"
    assert d is None


def test_parse_response_handles_bad_input():
    s, w, d = _parse_response("")
    assert s is None and w is None and d is None


def test_parse_response_definition_as_string():
    text = '{"summary": "s", "why": "w", "definition": "a plain string def"}'
    s, w, d = _parse_response(text)
    assert s == "s" and w == "w"
    assert d == {"text": "a plain string def", "references": []}


def test_parse_response_without_definition():
    text = '{"summary": "hello", "why": "because"}'
    s, w, d = _parse_response(text)
    assert s == "hello" and w == "because"
    assert d is None


def test_generate_blurb_skips_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert generate_blurb(word="mcp", candidates=[], evidence_articles=[]) is None


def test_attach_blurb_to_wotd_no_key_preserves_file(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    wotd_path = tmp_path / "2026-04-13.json"
    payload = {"date": "2026-04-13", "word": "mcp", "candidates": [], "evidence_article_ids": []}
    wotd_path.write_text(json.dumps(payload))
    ok = attach_blurb_to_wotd(wotd_path, evidence_articles=[])
    assert ok is False
    # File unchanged.
    assert json.loads(wotd_path.read_text()) == payload


BLURB = {"summary": "Gemini 4 Argon shipped.", "why": "Google launched it.", "definition": {"text": "A model.", "references": []}}


def _mock_openrouter(monkeypatch, handler) -> None:
    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.llm.httpx.Client", FakeClient)


def test_generate_blurb_asks_openrouter_when_its_key_is_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["url"] = str(request.url)
        sent["auth"] = request.headers["authorization"]
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(BLURB)}}]})

    _mock_openrouter(monkeypatch, handler)
    blurb = generate_blurb(
        word="Gemini 4 Argon",
        candidates=[],
        evidence_articles=[{"source_id": "s", "title": "t", "content_text": "Google released Gemini 4 Argon."}],
        openrouter_models=["google/gemma-4-31b-it:free"],
    )

    assert sent["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert sent["auth"] == "Bearer or-key"
    assert sent["body"]["model"] == "google/gemma-4-31b-it:free"
    assert "Google released Gemini 4 Argon." in sent["body"]["messages"][1]["content"]
    assert blurb["why"] == "Google launched it."
    assert blurb["model"] == "google/gemma-4-31b-it:free"


def test_generate_blurb_waits_out_a_free_tier_limit_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    replies = [
        httpx.Response(429, headers={"retry-after": "7"}, json={"error": "rate limited"}),
        httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(BLURB)}}]}),
    ]
    _mock_openrouter(monkeypatch, lambda request: replies.pop(0))
    waits: list[float] = []

    blurb = generate_blurb(
        word="mcp", candidates=[], evidence_articles=[],
        openrouter_models=["a:free"], sleep=waits.append,
    )

    assert blurb["model"] == "a:free"
    assert waits == [7.0]


def test_generate_blurb_moves_to_the_next_free_model_when_one_stays_limited(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "busy:free":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(BLURB)}}]})

    _mock_openrouter(monkeypatch, handler)
    blurb = generate_blurb(
        word="mcp", candidates=[], evidence_articles=[],
        openrouter_models=["busy:free", "idle:free"], sleep=lambda s: None,
    )

    assert blurb["model"] == "idle:free"


def test_generate_blurb_gives_up_quietly_when_every_model_refuses(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(429, json={"error": "rate limited"})

    _mock_openrouter(monkeypatch, handler)
    blurb = generate_blurb(
        word="mcp", candidates=[], evidence_articles=[],
        openrouter_models=["a:free", "b:free"], sleep=lambda s: None,
    )

    assert blurb is None
    assert calls == ["a:free"] * 3 + ["b:free"] * 3


def test_generate_blurb_does_not_retry_a_bad_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "bad")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": "no auth"})

    _mock_openrouter(monkeypatch, handler)
    assert generate_blurb(
        word="mcp", candidates=[], evidence_articles=[],
        openrouter_models=["a:free", "b:free"], sleep=lambda s: None,
    ) is None
    assert len(calls) == 1


def test_generate_blurb_moves_on_when_a_model_answers_with_broken_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        content = "sorry, no JSON" if model == "chatty:free" else json.dumps(BLURB)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    _mock_openrouter(monkeypatch, handler)
    blurb = generate_blurb(
        word="mcp", candidates=[], evidence_articles=[],
        openrouter_models=["chatty:free", "strict:free"], sleep=lambda s: None,
    )

    assert blurb["model"] == "strict:free"


def test_generate_blurb_never_sends_a_paid_model(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(BLURB)}}]})

    _mock_openrouter(monkeypatch, handler)
    blurb = generate_blurb(
        word="mcp", candidates=[], evidence_articles=[],
        openrouter_models=["anthropic/claude-sonnet-5.5", "openrouter/free"], sleep=lambda s: None,
    )

    assert sent == ["openrouter/free"]
    assert blurb["model"] == "openrouter/free"


def test_generate_blurb_moves_on_when_one_model_is_forbidden(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "harness-only:free":
            return httpx.Response(403, json={"error": "only available on agentic harnesses"})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(BLURB)}}]})

    _mock_openrouter(monkeypatch, handler)
    blurb = generate_blurb(
        word="mcp", candidates=[], evidence_articles=[],
        openrouter_models=["harness-only:free", "openrouter/free"], sleep=lambda s: None,
    )

    assert blurb["model"] == "openrouter/free"
