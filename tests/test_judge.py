"""Tests for the TypeSafe (Jev) candidate judge."""

from __future__ import annotations

import httpx
import pytest

from wotd.judge import JudgeError, Verdict, judge_candidates, rank


TERMS = ["kimi k3", "he breaks down", "claude"]
ARTICLES = [
    {"source_id": "codenewsletter", "title": "Moonshot drops Kimi K3", "snippet": "..."},
    {"source_id": "aibreakfast", "title": "Open Chinese model beats Fable 5", "snippet": "..."},
]
RECENT = ["Claude Tag ships today", "Gemini Omni lands"]


def _answers(values: dict[str, tuple[float, float, float, float, float]]) -> dict:
    """Build a System One response body for the given per-term signals."""
    out: dict[str, dict] = {}
    for i, term in enumerate(TERMS):
        form, spec, event, novelty, dominance = values[term]
        out[f"f{i:02d}"] = {"type": "noul", "noul": form}
        out[f"s{i:02d}"] = {"type": "score", "score": spec, "confidence": 0.9}
        out[f"e{i:02d}"] = {"type": "noul", "noul": event}
        out[f"n{i:02d}"] = {"type": "noul", "noul": novelty}
        out[f"d{i:02d}"] = {"type": "score", "score": dominance, "confidence": 0.9}
    return out


def _mock_client(monkeypatch, handler) -> None:
    original = httpx.Client

    class FakeClient(original):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    monkeypatch.setattr("wotd.judge.httpx.Client", FakeClient)


def test_judge_sends_state_and_five_questions_per_term(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": _answers(
                    {
                        "kimi k3": (0.9, 2.0, 0.9, 0.8, 2.4),
                        "he breaks down": (0.2, 0.1, 0.3, 0.3, 1.0),
                        "claude": (0.9, 1.8, 0.9, 0.4, 2.9),
                    }
                ),
                "usage": {"input_tokens": 1234, "output_tokens": 99},
            },
        )

    _mock_client(monkeypatch, handler)
    result = judge_candidates(
        terms=TERMS, articles=ARTICLES, recent_titles=RECENT, date="2026-07-17"
    )

    assert captured["model"] == "jev-latest"
    assert captured["state"]["date"] == "2026-07-17"
    assert len(captured["state"]["today"]) == 2
    assert captured["state"]["recent"] == RECENT
    assert len(captured["questions"]) == len(TERMS) * 5
    assert captured["questions"]["s00"]["type"] == "score"
    assert "kimi k3" in captured["questions"]["s00"]["instructions"]

    assert result.model == "jev-1.13.0"
    assert result.input_tokens == 1234
    assert result.verdicts["kimi k3"] == Verdict(
        form=0.9, specificity=2.0, event=0.9, novelty=0.8, dominance=2.4
    )


def test_rank_drops_fragments_and_prefers_the_newsworthy_term(monkeypatch):
    verdicts = {
        "kimi k3": Verdict(0.9, 2.0, 0.9, 0.8, 2.4),
        "he breaks down": Verdict(0.2, 0.1, 0.3, 0.3, 1.0),
        "claude": Verdict(0.9, 1.8, 0.9, 0.4, 2.9),
    }
    survivors = rank(verdicts)
    assert "he breaks down" not in survivors
    assert survivors[0] == "kimi k3"


def test_rank_prefers_the_fuller_name_on_a_near_tie():
    verdicts = {
        "claude": Verdict(0.9, 1.9, 0.95, 0.83, 2.99),
        "claude tag": Verdict(0.9, 1.99, 0.96, 0.84, 2.99),
    }
    assert rank(verdicts)[0] == "claude tag"


def test_rank_returns_empty_when_nothing_survives():
    verdicts = {"nitter net": Verdict(0.3, 0.1, 0.1, 0.9, 0.2)}
    assert rank(verdicts) == []


def test_rank_drops_a_real_term_the_day_does_not_cover():
    """A good AI word nobody wrote about today is not the word of the day."""
    verdicts = {"baseline": Verdict(0.85, 1.51, 0.17, 0.14, 0.02)}
    assert rank(verdicts) == []


def test_judge_raises_without_a_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JudgeError, match="TYPESAFE_API_KEY"):
        judge_candidates(
            terms=TERMS, articles=ARTICLES, recent_titles=RECENT, date="2026-07-17"
        )


def test_judge_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(429, text="slow down")

    _mock_client(monkeypatch, handler)
    with pytest.raises(JudgeError, match="429"):
        judge_candidates(
            terms=TERMS,
            articles=ARTICLES,
            recent_titles=RECENT,
            date="2026-07-17",
            sleep=lambda _: None,
        )
    assert len(attempts) == 3


def test_judge_retries_a_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(529, text="overloaded")
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": _answers(
                    {
                        "kimi k3": (0.9, 2.0, 0.9, 0.8, 2.4),
                        "he breaks down": (0.2, 0.1, 0.3, 0.3, 1.0),
                        "claude": (0.9, 1.8, 0.9, 0.4, 2.9),
                    }
                ),
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    _mock_client(monkeypatch, handler)
    result = judge_candidates(
        terms=TERMS,
        articles=ARTICLES,
        recent_titles=RECENT,
        date="2026-07-17",
        sleep=lambda _: None,
    )
    assert calls["n"] == 2
    assert result.verdicts["kimi k3"].specificity == 2.0


def test_judge_does_not_retry_a_bad_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(401, text="unauthorized")

    _mock_client(monkeypatch, handler)
    with pytest.raises(JudgeError, match="401"):
        judge_candidates(
            terms=TERMS,
            articles=ARTICLES,
            recent_titles=RECENT,
            date="2026-07-17",
            sleep=lambda _: None,
        )
    assert len(attempts) == 1


def test_judge_raises_when_an_answer_is_missing(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    answers = _answers(
        {
            "kimi k3": (0.9, 2.0, 0.9, 0.8, 2.4),
            "he breaks down": (0.2, 0.1, 0.3, 0.3, 1.0),
            "claude": (0.9, 1.8, 0.9, 0.4, 2.9),
        }
    )
    answers.pop("d02")
    _mock_client(
        monkeypatch,
        lambda request: httpx.Response(200, json={"model": "jev", "answers": answers}),
    )
    with pytest.raises(JudgeError, match="d02"):
        judge_candidates(
            terms=TERMS, articles=ARTICLES, recent_titles=RECENT, date="2026-07-17"
        )


def test_judge_returns_nothing_for_an_empty_pool(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    result = judge_candidates(
        terms=[], articles=ARTICLES, recent_titles=RECENT, date="2026-07-17"
    )
    assert result.verdicts == {}
    assert result.input_tokens == 0
