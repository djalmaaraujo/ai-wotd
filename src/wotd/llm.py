"""LLM client: daily summary + why-it-trended blurb, through OpenRouter's free models.

No-op when OPENROUTER_API_KEY is unset so local dev and CI without the secret
still succeed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx

logger = logging.getLogger(__name__)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = (
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "thinkingmachines/inkling:free",
    "openrouter/free",
)
# The blurb must never cost money: only `:free` variants and OpenRouter's
# free router are ever sent.
FREE_ROUTER = "openrouter/free"
OPENROUTER_TIMEOUT = 120.0
ATTEMPTS_PER_MODEL = 3
BACKOFF_SECONDS = 20.0
MAX_WAIT_SECONDS = 90.0
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_TOKENS = 1200


SUMMARY_PROMPT = """You are writing for an AI industry daily digest.

Given the list of the top trending AI terms for today along with headlines and
short excerpts from the articles that mention them, write THREE things:

1. summary — about 150 words. A running narrative of what is happening in AI
   today as seen through this corpus. Mention 2-4 of the top terms, with
   concrete references (companies, products, people). No hype, no marketing,
   no preamble. Start with a declarative sentence.
2. why — 2–3 sentences that specifically explain why the chosen WORD OF THE
   DAY trended today, grounded in the evidence excerpts. Be concrete.
3. definition — a concise definition (2-4 sentences) of the WORD OF THE DAY
   in the context of AI. Explain what the term means, how it is used in the
   AI industry, and why it matters. If the term is a product or company name,
   describe what it does and its role in the AI ecosystem. Ground the
   definition in publicly available knowledge. Include a "references" list
   of 1-3 authoritative URLs (official docs, Wikipedia, seminal papers) that
   a reader could follow to learn more. Each reference should be a plain URL
   string.

Return a single JSON object with keys "summary", "why", and "definition",
where "definition" is an object with keys "text" (the definition string) and
"references" (an array of URL strings). Nothing else.
"""


def _build_user_message(word: str, candidates: list[dict], evidence: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"WORD OF THE DAY: {word}")
    lines.append("")
    lines.append("Top candidates (term, tf_today, df_today):")
    for c in candidates[:10]:
        lines.append(f"  - {c['term']} (tf={c['tf_today']}, df={c['df_today']})")
    lines.append("")
    lines.append("Evidence articles:")
    for art in evidence[:8]:
        lines.append(f"  - [{art.get('source_id','?')}] {art.get('title','(no title)')}")
        lines.append(f"    url: {art.get('url','')}")
        snippet = art.get("content_text") or art.get("snippet") or ""
        if snippet:
            lines.append(f"    excerpt: {snippet[:800]}")
    return "\n".join(lines)


def _retry_wait(response: httpx.Response | None, attempt: int) -> float:
    try:
        asked = float(response.headers.get("retry-after", "")) if response is not None else 0.0
    except ValueError:
        asked = 0.0
    return min(asked or BACKOFF_SECONDS * (attempt + 1), MAX_WAIT_SECONDS)


class _Refused(Exception):
    """OpenRouter turned the request down for a reason no retry will fix."""


def _ask_openrouter(system: str, user: str, model: str, key: str, *, sleep=time.sleep) -> str | None:
    """Return the model's text, retrying a free-tier rate limit with backoff.

    Raises `_Refused` on an error that is the same for every model (bad key,
    no credit), so the caller stops instead of walking the whole list.
    """
    for attempt in range(ATTEMPTS_PER_MODEL):
        response = None
        try:
            with httpx.Client(timeout=OPENROUTER_TIMEOUT) as client:
                response = client.post(
                    OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {key}", "X-Title": "ai-wotd"},
                    json={
                        "model": model,
                        "temperature": 0.2,
                        "max_tokens": MAX_TOKENS,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    },
                )
        except httpx.HTTPError as exc:
            logger.warning("llm: %s request failed: %s", model, exc)
        else:
            if response.status_code == 200:
                text = _openrouter_text(response)
                if text:
                    return text
                logger.warning("llm: %s answered with no text", model)
            elif response.status_code in RETRYABLE_STATUS:
                logger.warning("llm: %s returned %s", model, response.status_code)
            else:
                raise _Refused(f"OpenRouter returned {response.status_code}: {response.text[:200]}")
        if attempt < ATTEMPTS_PER_MODEL - 1:
            sleep(_retry_wait(response, attempt))
    return None


def _openrouter_text(response: httpx.Response) -> str:
    try:
        choices = response.json().get("choices") or []
        return (choices[0]["message"].get("content") or "").strip() if choices else ""
    except (ValueError, KeyError, TypeError, AttributeError):
        return ""


def generate_blurb(
    *,
    word: str,
    candidates: list[dict],
    evidence_articles: list[dict],
    api_key: str | None = None,
    openrouter_models: Sequence[str] | None = None,
    sleep=time.sleep,
) -> dict | None:
    """Ask an LLM for {summary, why, definition, model, generated_at}.

    Returns None (and logs) when no key is set or every attempt fails.
    """
    if not word:
        logger.info("llm: skipped (no WOTD word)")
        return None
    user_msg = _build_user_message(word, candidates, evidence_articles)

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        logger.info("llm: skipped (no OPENROUTER_API_KEY)")
        return None
    # Free models share a small rate limit and sometimes break the JSON,
    # so each one is tried in turn until one gives a usable blurb.
    for model in _free_only(openrouter_models or OPENROUTER_MODELS):
        try:
            text = _ask_openrouter(SUMMARY_PROMPT, user_msg, model, key, sleep=sleep)
        except _Refused as exc:
            logger.warning("llm: %s", exc)
            return None
        blurb = _blurb_from(text, model) if text else None
        if blurb:
            return blurb
    return None


def _free_only(models: Sequence[str]) -> list[str]:
    free = [m for m in models if m.endswith(":free") or m == FREE_ROUTER]
    for model in models:
        if model not in free:
            logger.warning("llm: skipping %s; only free OpenRouter models are allowed", model)
    return free


def _blurb_from(text: str, model: str) -> dict | None:
    summary, why, definition = _parse_response(text)
    if not summary or not why:
        logger.warning("llm: could not parse %s's response; raw=%r", model, text[:200])
        return None
    result = {
        "summary": summary,
        "why": why,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if definition:
        result["definition"] = definition
    return result


def _parse_response(text: str) -> tuple[str | None, str | None, dict | None]:
    """Extract `summary`, `why`, and `definition` from the model output.

    Accepts either a JSON object or a code-fenced JSON object.
    Returns (summary, why, definition) where definition is
    ``{"text": "...", "references": [...]}`` or None.
    """
    if not text:
        return None, None, None

    # Strip markdown fences if present.
    candidate = text.strip()
    if candidate.startswith("```"):
        # remove first and last fence lines
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        # Fallback: look for a {...} block.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None, None, None
        else:
            return None, None, None

    summary = obj.get("summary") if isinstance(obj, dict) else None
    why = obj.get("why") if isinstance(obj, dict) else None
    if isinstance(summary, str):
        summary = summary.strip() or None
    else:
        summary = None
    if isinstance(why, str):
        why = why.strip() or None
    else:
        why = None

    definition = None
    raw_def = obj.get("definition") if isinstance(obj, dict) else None
    if isinstance(raw_def, dict):
        def_text = raw_def.get("text")
        if isinstance(def_text, str) and def_text.strip():
            refs = raw_def.get("references", [])
            if not isinstance(refs, list):
                refs = []
            refs = [str(r) for r in refs if isinstance(r, str) and r.strip()]
            definition = {"text": def_text.strip(), "references": refs}
    elif isinstance(raw_def, str) and raw_def.strip():
        # Tolerate a plain string (no references).
        definition = {"text": raw_def.strip(), "references": []}

    return summary, why, definition


def attach_blurb_to_wotd(
    wotd_path: Path,
    evidence_articles: list[dict],
    api_key: str | None = None,
    openrouter_models: Sequence[str] | None = None,
) -> bool:
    """Load `wotd_path`, call the LLM, and persist the blurb back.

    Returns True if a blurb was written, False otherwise.
    """
    if not wotd_path.exists():
        return False
    with open(wotd_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    word = payload.get("word")
    if not word:
        return False

    blurb = generate_blurb(
        word=payload.get("label") or word,
        candidates=payload.get("candidates", []),
        evidence_articles=evidence_articles,
        api_key=api_key,
        openrouter_models=openrouter_models,
    )
    if not blurb:
        return False

    payload["llm"] = blurb
    with open(wotd_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return True
