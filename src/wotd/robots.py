"""robots.txt and X-Robots-Tag, which the README promises publishers we honour.

One fetch per scheme, host and port per run, cached. A host whose rules cannot be
read answers `unknown`, which is not consent: the caller skips the source for this
run rather than ingesting it half-read, so the next run can try again.

Nothing here raises. This sits inside the fetch loop, and an exception escaping it
aborts the whole run and throws away every cursor.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)

ROBOTS_TIMEOUT = 10.0

ALLOW = "allow"
DENY = "deny"
UNKNOWN = "unknown"

_NOINDEX = {"noindex", "none"}


def blocked_by_header(headers) -> bool:
    """True when a response asks not to be indexed.

    Handles the agent-scoped form (`X-Robots-Tag: ai-wotd: noindex`) by reading a
    directive that names an agent as applying only to that agent.
    """
    for key, value in dict(headers).items():
        if key.lower() != "x-robots-tag":
            continue
        for directive in str(value).split(","):
            directive = directive.strip().lower()
            agent, _, scoped = directive.partition(":")
            if scoped:
                if agent.strip() in ("*", "ai-wotd") and scoped.strip() in _NOINDEX:
                    return True
            elif directive in _NOINDEX:
                return True
    return False


class RobotsCache:
    """Per-origin robots.txt rules, fetched at most once."""

    def __init__(self, user_agent: str, timeout: float = ROBOTS_TIMEOUT) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._origins: dict[tuple[str, str, int | None], RobotFileParser | None] = {}

    def _origin(self, url: str) -> tuple[str, str, int | None] | None:
        try:
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https"):
                return None
            host = parts.hostname
            port = parts.port
        except (ValueError, UnicodeError):
            return None
        if not host or not host.strip():
            return None
        return (parts.scheme, host.lower(), port)

    def _parser_for(
        self, origin: tuple[str, str, int | None]
    ) -> RobotFileParser | None:
        if origin in self._origins:
            return self._origins[origin]

        scheme, host, port = origin
        netloc = f"{host}:{port}" if port else host
        url = f"{scheme}://{netloc}/robots.txt"
        parser: RobotFileParser | None = RobotFileParser()
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url, headers={"User-Agent": self.user_agent})
        except Exception as exc:
            logger.warning("robots: %s unreachable (%s); the host stays unread", url, exc)
            parser = None
        else:
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            elif 400 <= response.status_code < 500:
                parser.parse([])
            else:
                logger.warning(
                    "robots: %s returned %s; the host stays unread",
                    url,
                    response.status_code,
                )
                parser = None

        self._origins[origin] = parser
        return parser

    def status(self, url: str) -> str:
        """`allow`, `deny`, or `unknown` when the rules could not be read."""
        origin = self._origin(url)
        if origin is None:
            return DENY
        parser = self._parser_for(origin)
        if parser is None:
            return UNKNOWN
        try:
            return ALLOW if parser.can_fetch(self.user_agent, url) else DENY
        except Exception as exc:
            logger.warning("robots: could not apply rules to %s (%s)", url, exc)
            return DENY

    def allowed(self, url: str) -> bool:
        return self.status(url) == ALLOW
