"""TypeSafe (Jev) judge: typed judgments about candidate terms.

The deterministic scorer in `wotd.py` says which terms are unusual today. It
cannot say whether a term is a real AI thing, whether the string is even a name,
or whether today's coverage is about it. This module asks those questions and
returns numbers; the caller composes them into a decision.

Unlike the reranker it replaces, every failure raises. A judge that goes quiet
must be visible, not swallowed.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import httpx

logger = logging.getLogger(__name__)


API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 60.0
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})

MAX_ARTICLES = 60
MAX_RECENT_TITLES = 60

SPECIFICITY_MIN = 1.5
FORM_MIN = 0.8
DOMINANCE_MIN = 1.5
EVENT_MIN = 0.5
NOVELTY_WEIGHT = 0.5
DOMINANCE_LEVELS = 3.0
TIE_MARGIN = 0.05

FORM = (
    "Is {term!r} a well-formed name or noun phrase that could stand alone as a "
    "headline term? Answer no if it is a broken sentence fragment, a stray "
    "preposition or verb phrase, website or newsletter boilerplate (subscribe, "
    "read more, sponsored by), a social-media handle, a domain name or URL "
    "fragment, or a text-extraction artifact."
)
SPECIFICITY = "How specific is {term!r} as a term from the AI/ML field?"
SPECIFICITY_LEVELS = [
    "Not an AI term at all: ordinary English, news vocabulary, or unrelated to AI",
    "Vague AI vocabulary: a broad word such as 'models', 'ai', 'research', or a "
    "bare company name with nothing specific attached",
    "A specific AI thing: a named model, product, benchmark, architecture, "
    "technique, or piece of field jargon",
]
EVENT = (
    "Do the articles in `today` report a new event about {term!r} - a launch, "
    "release, suspension, incident, outage, acquisition, benchmark result, policy "
    "decision or paper - as opposed to {term!r} only being mentioned in passing or "
    "used as background vocabulary?"
)
NOVELTY = (
    "`recent` holds headlines from the seven days before today. Compared with "
    "`recent`, has {term!r} become newly prominent in `today`? Answer no if "
    "`recent` already covers it just as much, and no if it is ordinary vocabulary "
    "that is always present."
)
DOMINANCE = "How much of today's coverage in `today` is about {term!r}?"
DOMINANCE_LEGEND = [
    "Not covered: the phrase does not name anything today's articles are reporting on",
    "A single passing mention inside one article",
    "One clear story among several the day covers",
    "The main story of the day: several articles or several sources return to it",
]


class JudgeError(RuntimeError):
    """The judge could not produce a verdict."""


@dataclass(frozen=True)
class Verdict:
    form: float
    specificity: float
    event: float
    novelty: float
    dominance: float

    def to_dict(self) -> dict:
        return {
            "form": round(self.form, 3),
            "specificity": round(self.specificity, 3),
            "event": round(self.event, 3),
            "novelty": round(self.novelty, 3),
            "dominance": round(self.dominance, 3),
        }


@dataclass(frozen=True)
class Judgment:
    model: str
    verdicts: dict[str, Verdict]
    input_tokens: int


def _state(articles: Iterable[dict], recent_titles: Sequence[str], date: str) -> dict:
    today = [
        {
            "source": a.get("source_id") or a.get("source"),
            "title": a.get("title"),
            "snippet": a.get("snippet"),
        }
        for a in list(articles)[:MAX_ARTICLES]
    ]
    return {"date": date, "today": today, "recent": list(recent_titles)[:MAX_RECENT_TITLES]}


def _questions(terms: Sequence[str]) -> dict:
    questions: dict[str, dict] = {}
    for i, term in enumerate(terms):
        questions[f"f{i:02d}"] = {
            "type": "noul",
            "instructions": FORM.format(term=term),
        }
        questions[f"s{i:02d}"] = {
            "type": "score",
            "instructions": SPECIFICITY.format(term=term),
            "criteria": SPECIFICITY_LEVELS,
        }
        questions[f"e{i:02d}"] = {
            "type": "noul",
            "instructions": EVENT.format(term=term),
        }
        questions[f"n{i:02d}"] = {
            "type": "noul",
            "instructions": NOVELTY.format(term=term),
        }
        questions[f"d{i:02d}"] = {
            "type": "score",
            "instructions": DOMINANCE.format(term=term),
            "criteria": DOMINANCE_LEGEND,
        }
    return questions


def _read(answers: dict, key: str, field: str) -> float:
    answer = answers.get(key)
    if not isinstance(answer, dict) or field not in answer:
        raise JudgeError(f"judge: answer {key!r} missing from the response")
    try:
        return float(answer[field])
    except (TypeError, ValueError) as exc:
        raise JudgeError(f"judge: answer {key!r} is not a number") from exc


def _post_with_retries(payload: dict, key: str, timeout: float, *, sleep=time.sleep):
    """POST once per attempt, backing off on the statuses TypeSafe asks us to retry."""
    last: str = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    API_URL,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.HTTPError as exc:
            last = f"request failed: {exc}"
        else:
            if response.status_code == 200:
                return response
            last = f"TypeSafe returned {response.status_code}: {response.text[:200]}"
            if response.status_code not in RETRYABLE_STATUS:
                break
        if attempt < MAX_ATTEMPTS - 1:
            logger.warning("judge: attempt %d failed (%s); retrying", attempt + 1, last)
            sleep(BACKOFF_SECONDS * (attempt + 1))
    raise JudgeError(f"judge: {last}")


def judge_candidates(
    *,
    terms: Sequence[str],
    articles: Iterable[dict],
    recent_titles: Sequence[str],
    date: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    sleep=time.sleep,
) -> Judgment:
    """Judge every candidate term in a single request. Raises `JudgeError`."""
    key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise JudgeError("judge: TYPESAFE_API_KEY is not set")
    if not terms:
        return Judgment(model=model, verdicts={}, input_tokens=0)

    payload = {
        "model": model,
        "state": _state(articles, recent_titles, date),
        "questions": _questions(terms),
    }

    response = _post_with_retries(payload, key, timeout, sleep=sleep)

    try:
        body = response.json()
    except ValueError as exc:
        raise JudgeError("judge: response was not JSON") from exc

    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise JudgeError("judge: response carried no answers")

    verdicts = {
        term: Verdict(
            form=_read(answers, f"f{i:02d}", "noul"),
            specificity=_read(answers, f"s{i:02d}", "score"),
            event=_read(answers, f"e{i:02d}", "noul"),
            novelty=_read(answers, f"n{i:02d}", "noul"),
            dominance=_read(answers, f"d{i:02d}", "score"),
        )
        for i, term in enumerate(terms)
    }

    usage = body.get("usage") or {}
    judgment = Judgment(
        model=str(body.get("model") or model),
        verdicts=verdicts,
        input_tokens=int(usage.get("input_tokens") or 0),
    )
    logger.info(
        "judge: %s judged %d candidates (%d input tokens)",
        judgment.model,
        len(verdicts),
        judgment.input_tokens,
    )
    return judgment


def _newsworthiness(verdict: Verdict) -> float:
    return NOVELTY_WEIGHT * verdict.novelty + verdict.dominance / DOMINANCE_LEVELS


def rank(
    verdicts: dict[str, Verdict],
    *,
    specificity_min: float = SPECIFICITY_MIN,
    form_min: float = FORM_MIN,
    dominance_min: float = DOMINANCE_MIN,
    event_min: float = EVENT_MIN,
) -> list[str]:
    """Survivors of the gate, most newsworthy first.

    A word has to be a real AI term (`specificity`), a usable string (`form`),
    something today's articles actually cover (`dominance`) and something that
    happened (`event`). Terms within `TIE_MARGIN` of each other are ordered by
    how fully they name the thing, so "claude tag" wins over "claude" on the day
    it launched.
    """
    survivors = [
        term
        for term, v in verdicts.items()
        if v.specificity >= specificity_min
        and v.form >= form_min
        and v.dominance >= dominance_min
        and v.event >= event_min
    ]
    survivors.sort(
        key=lambda term: (
            -round(_newsworthiness(verdicts[term]) / TIE_MARGIN),
            -len(term.split()),
            -verdicts[term].specificity,
            term,
        )
    )
    return survivors
