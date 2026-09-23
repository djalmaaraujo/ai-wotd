"""robots.txt and X-Robots-Tag, which the README promises publishers we honour.

One fetch per host per run, cached. A host that cannot be read is treated as
closed: an unreachable robots.txt is unknown intent, and unknown is not consent.
"""

from __future__ import annotations

import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)

ROBOTS_TIMEOUT = 10.0


def blocked_by_header(headers) -> bool:
    """True when a response asks not to be indexed."""
    for key, value in dict(headers).items():
        if key.lower() == "x-robots-tag":
            directives = {d.strip().lower() for d in str(value).split(",")}
            if "noindex" in directives or "none" in directives:
                return True
    return False


class RobotsCache:
    """Per-host robots.txt rules, fetched at most once."""

    def __init__(self, user_agent: str, timeout: float = ROBOTS_TIMEOUT) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._hosts: dict[str, RobotFileParser | None] = {}

    def _parser_for(self, scheme: str, host: str) -> RobotFileParser | None:
        if host in self._hosts:
            return self._hosts[host]

        parser: RobotFileParser | None = RobotFileParser()
        url = f"{scheme}://{host}/robots.txt"
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url, headers={"User-Agent": self.user_agent})
        except httpx.HTTPError as exc:
            logger.info("robots: %s unreachable (%s); treating the host as closed", url, exc)
            parser = None
        else:
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            elif 400 <= response.status_code < 500:
                parser.parse([])
            else:
                logger.info(
                    "robots: %s returned %s; treating the host as closed",
                    url,
                    response.status_code,
                )
                parser = None

        self._hosts[host] = parser
        return parser

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return False
        parser = self._parser_for(parts.scheme, parts.netloc)
        if parser is None:
            return False
        return parser.can_fetch(self.user_agent, url)
