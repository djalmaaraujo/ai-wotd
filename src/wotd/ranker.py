"""Groq-assisted AI-domain reranker for WOTD candidates.

The deterministic scorer in wotd.py produces a long list of high-TF
candidates. Many of them are generic news vocabulary or leftover noise
("market", "companies", "yesterday"). This module takes the top N of
those and asks a fast free-tier Groq model which ones are actually
AI/ML domain terms — one call per day, not per article.

Activated only when `GROQ_API_KEY` is set. Falls back silently
otherwise: the deterministic top pick stays the WOTD.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an AI/ML domain expert helping pick a "Word of the Day" from trending terms in AI industry news.

You receive a list of candidate terms extracted from today's articles. Your job: rank them by how well each qualifies as a genuine AI/ML/agents/LLMs domain term — a concept, technique, architecture, product, model family, benchmark, or emerging jargon that a reader of AI news would recognize as substantive.

Rules (hard):
- REJECT generic English or news vocabulary (e.g. "companies", "users", "market", "report", "new", "today", "yesterday").
- REJECT file-format artifacts (e.g. "obj", "endobj", "xref", "q q", "n n", any single-letter pair).
- REJECT bare company names that appear every day ("openai", "google", "microsoft") unless paired with a specific product ("openai o3", "google gemini").
- PREFER concrete technical concepts: "context window", "mcp", "rag", "rlhf", "agentic", "tool use", "reasoning model".
- PREFER emerging or contested ideas that are trending today over commodity terms.
- PREFER multi-word n-grams over single words when both refer to the same concept.

Return STRICT JSON: {"top": ["term1", "term2", "term3", "term4", "term5"]}. Exactly 5 terms, ordered by relevance descending. Copy the candidate strings VERBATIM from the input — do not paraphrase, rewrite, or invent terms."""


def _build_user_message(candidates: Iterable[dict]) -> str:
    lines = ["Candidate terms for today's Word of the Day:", ""]
    for c in candidates:
        term = c.get("term", "")
        tf = c.get("tf_today", 0)
        df = c.get("df_today", 0)
        lines.append(f"- {term!r} (tf={tf}, df={df})")
    return "\n".join(lines)


def _parse_top(text: str) -> list[str] | None:
    """Extract ['term1', ...] from the LLM response. Tolerant parser."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    top = obj.get("top") if isinstance(obj, dict) else None
    if not isinstance(top, list):
        return None
    clean = [str(t).strip() for t in top if isinstance(t, str) and t.strip()]
    return clean or None


def rerank_candidates(
    candidates: list[dict],
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    top_n_input: int = 40,
    timeout: float = 20.0,
) -> list[str] | None:
    """Return an AI-domain-ranked list of term strings, or None on failure.

    - `candidates` is the deterministic candidate list from wotd.score_terms
      (each is a dict with at least `term`, `tf_today`, `df_today`).
    - Sends the top `top_n_input` to Groq (one call total).
    - Returns the LLM's top choices, copied verbatim from the input. Any
      term the LLM invented (not present in the input) is dropped — that's
      why the prompt enforces verbatim copies.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        logger.info("rerank: skipped (no GROQ_API_KEY)")
        return None
    if not candidates:
        return None

    head = candidates[:top_n_input]
    user_msg = _build_user_message(head)
    valid_terms = {c.get("term") for c in head if c.get("term")}

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
    except Exception as exc:
        logger.warning("rerank: Groq request failed: %s", exc)
        return None

    if resp.status_code != 200:
        logger.warning(
            "rerank: Groq returned %s: %s",
            resp.status_code,
            resp.text[:200],
        )
        return None

    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("rerank: unexpected Groq shape: %s", exc)
        return None

    top = _parse_top(text)
    if not top:
        logger.warning("rerank: could not parse Groq response; raw=%r", text[:200])
        return None

    # Keep only terms that appeared in the input (anti-hallucination guard).
    filtered = [t for t in top if t in valid_terms]
    if not filtered:
        logger.warning(
            "rerank: Groq returned none of the input terms verbatim; raw=%r",
            text[:200],
        )
        return None

    logger.info("rerank: Groq picked top-%d: %s", len(filtered), filtered)
    return filtered
