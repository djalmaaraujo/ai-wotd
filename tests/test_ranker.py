"""Tests for the Groq-assisted WOTD reranker."""

from __future__ import annotations

import httpx
import pytest

from wotd.ranker import _parse_top, rerank_candidates


CANDIDATES = [
    {"term": "obj", "tf_today": 4000, "df_today": 4},
    {"term": "mcp", "tf_today": 300, "df_today": 8},
    {"term": "context window", "tf_today": 80, "df_today": 5},
    {"term": "companies", "tf_today": 250, "df_today": 9},
    {"term": "agentic", "tf_today": 150, "df_today": 7},
]


def test_parse_top_plain_json():
    assert _parse_top('{"top": ["a", "b", "c"]}') == ["a", "b", "c"]


def test_parse_top_handles_fenced_and_prose():
    assert _parse_top('```json\n{"top":["x"]}\n```') == ["x"]
    assert _parse_top('prefix {"top": ["y"]} suffix') == ["y"]
    assert _parse_top("nope") is None
    assert _parse_top("") is None


def test_rerank_skips_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert rerank_candidates(CANDIDATES) is None


def test_rerank_calls_groq_and_filters_hallucinations(monkeypatch):
    """Terms not present in the input get dropped from the output."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        # Groq returns a valid shape but includes one hallucinated term.
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"top": ["mcp", "agentic", "context window", "hallucinated_term"]}'
                        }
                    }
                ]
            },
        )

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.ranker.httpx.Client", FakeClient)

    result = rerank_candidates(CANDIDATES)
    assert result == ["mcp", "agentic", "context window"]
    # The hallucinated term got filtered out.
    assert "hallucinated_term" not in result


def test_rerank_returns_none_on_http_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.ranker.httpx.Client", FakeClient)

    assert rerank_candidates(CANDIDATES) is None


def test_rerank_returns_none_on_unparseable_response(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "I am not JSON."}}]},
        )

    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.ranker.httpx.Client", FakeClient)

    assert rerank_candidates(CANDIDATES) is None
