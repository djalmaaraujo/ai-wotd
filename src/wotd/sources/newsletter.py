"""Newsletter adapter — an RSS adapter that knows about platform quirks."""

from __future__ import annotations

from typing import ClassVar

from .rss import RssAdapter


class NewsletterAdapter(RssAdapter):
    type: ClassVar[str] = "newsletter"
