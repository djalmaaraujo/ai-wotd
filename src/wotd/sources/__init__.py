"""Source adapter registry."""

from __future__ import annotations

from .base import SourceAdapter  # noqa: F401
from .rss import RssAdapter
from .newsletter import NewsletterAdapter
from .twitter import TwitterAdapter


_REGISTRY: dict[str, type[SourceAdapter]] = {
    "rss": RssAdapter,
    "newsletter": NewsletterAdapter,
    "twitter": TwitterAdapter,
}


def get_adapter(type_name: str) -> SourceAdapter:
    cls = _REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown source type: {type_name!r}")
    return cls()
