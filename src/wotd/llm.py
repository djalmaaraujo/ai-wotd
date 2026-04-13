"""Anthropic client — daily summary + why-it-trended blurb.

No-op when ANTHROPIC_API_KEY is unset so local dev and CI without the secret
still succeed.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


SUMMARY_PROMPT = """You are writing for an AI industry daily digest.

Given the list of the top trending AI terms for today along with headlines and
short excerpts from the articles that mention them, write TWO things:

1. summary — about 150 words. A running narrative of what is happening in AI
   today as seen through this corpus. Mention 2-4 of the top terms, with
   concrete references (companies, products, people). No hype, no marketing,
   no preamble. Start with a declarative sentence.
2. why — 2–3 sentences that specifically explain why the chosen WORD OF THE
   DAY trended today, grounded in the evidence excerpts. Be concrete.

Return a single JSON object with keys "summary" and "why", and nothing else.
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
        snippet = art.get("snippet") or art.get("content_text") or ""
        if snippet:
            lines.append(f"    excerpt: {snippet[:800]}")
    return "\n".join(lines)


def generate_blurb(
    *,
    word: str,
    candidates: list[dict],
    evidence_articles: list[dict],
    model: str = "claude-sonnet-4-5",
    api_key: str | None = None,
) -> dict | None:
    """Call Anthropic Messages API; return {summary, why, model, generated_at}.

    Returns None (and logs) if the API key is missing or the SDK import fails.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.info("llm: skipped (no ANTHROPIC_API_KEY)")
        return None
    if not word:
        logger.info("llm: skipped (no WOTD word)")
        return None

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("llm: anthropic SDK not installed; skipping")
        return None

    client = Anthropic(api_key=key)
    user_msg = _build_user_message(word, candidates, evidence_articles)

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=800,
            temperature=0.2,
            system=SUMMARY_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as exc:  # network / API errors — don't fail the pipeline
        logger.warning("llm: Anthropic call failed: %s", exc)
        return None

    text = "".join(
        block.text for block in resp.content if getattr(block, "type", None) == "text"
    ).strip()

    summary, why = _parse_response(text)
    if not summary or not why:
        logger.warning("llm: could not parse response; raw=%r", text[:200])
        return None

    return {
        "summary": summary,
        "why": why,
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse_response(text: str) -> tuple[str | None, str | None]:
    """Extract `summary` and `why` from the model output.

    Accepts either a JSON object or a code-fenced JSON object.
    """
    if not text:
        return None, None

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
                return None, None
        else:
            return None, None

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
    return summary, why


def attach_blurb_to_wotd(
    wotd_path: Path,
    evidence_articles: list[dict],
    model: str = "claude-sonnet-4-5",
    api_key: str | None = None,
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
        word=word,
        candidates=payload.get("candidates", []),
        evidence_articles=evidence_articles,
        model=model,
        api_key=api_key,
    )
    if not blurb:
        return False

    payload["llm"] = blurb
    with open(wotd_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return True
