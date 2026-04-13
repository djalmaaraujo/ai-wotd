"""Configuration: load sources.yml + env overrides, expose paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_USER_AGENT = "ai-wotd/1.0 (+https://github.com/djalmaaraujo/ai-wotd)"


@dataclass
class Paths:
    root: Path
    data: Path
    articles: Path
    index: Path
    stats: Path
    wotd: Path
    parquet: Path
    docs: Path
    templates: Path
    fulltext_cache: Path

    @classmethod
    def from_root(cls, root: Path, fulltext_cache: Path | None = None) -> "Paths":
        root = Path(root)
        data = root / "data"
        return cls(
            root=root,
            data=data,
            articles=data / "articles",
            index=data / "index",
            stats=data / "stats",
            wotd=data / "wotd",
            parquet=data / "parquet",
            docs=root / "docs",
            templates=root / "templates",
            fulltext_cache=fulltext_cache
            or Path(os.environ.get("WOTD_FULLTEXT_CACHE_DIR", root / ".cache/wotd/fulltext")),
        )

    def ensure(self) -> None:
        for p in [
            self.data,
            self.articles,
            self.index,
            self.stats,
            self.wotd,
            self.parquet,
            self.docs,
            self.fulltext_cache,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    baseline_days: int = 30
    max_articles_per_source: int = 50
    user_agent: str = DEFAULT_USER_AGENT
    llm_model: str = "claude-sonnet-4-5"
    linkfollow_max_per_issue: int = 10
    anthropic_api_key: str | None = None
    nitter_instances: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            baseline_days=int(os.environ.get("WOTD_BASELINE_DAYS", "30")),
            max_articles_per_source=int(
                os.environ.get("WOTD_MAX_ARTICLES_PER_SOURCE", "50")
            ),
            user_agent=os.environ.get("WOTD_USER_AGENT", DEFAULT_USER_AGENT),
            llm_model=os.environ.get("WOTD_LLM_MODEL", "claude-sonnet-4-5"),
            linkfollow_max_per_issue=int(
                os.environ.get("WOTD_LINKFOLLOW_MAX_PER_ISSUE", "10")
            ),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
            nitter_instances=[
                s.strip()
                for s in os.environ.get("WOTD_NITTER_INSTANCES", "").split(",")
                if s.strip()
            ],
        )


def load_sources(path: Path) -> list[dict]:
    """Parse sources.yml and return the `sources` list."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if data.get("version") != 1:
        raise ValueError(f"Unsupported sources.yml version: {data.get('version')!r}")
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        raise ValueError("sources.yml: 'sources' must be a list")
    return sources
