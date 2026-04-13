"""Source adapter protocol + RawItem dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Iterable, Protocol

from ..corpus import RawItem  # re-exported


@dataclass
class Cursor:
    last_guid: str | None = None
    last_fetched_at: str | None = None
    etag: str | None = None
    last_modified: str | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "Cursor":
        d = d or {}
        return cls(
            last_guid=d.get("last_guid"),
            last_fetched_at=d.get("last_fetched_at"),
            etag=d.get("etag"),
            last_modified=d.get("last_modified"),
        )

    def to_dict(self) -> dict:
        return {
            "last_guid": self.last_guid,
            "last_fetched_at": self.last_fetched_at,
            "etag": self.etag,
            "last_modified": self.last_modified,
        }


class SourceAdapter(Protocol):
    type: ClassVar[str]

    def fetch(
        self,
        source: dict,
        cursor: Cursor,
        *,
        user_agent: str,
        max_items: int,
    ) -> Iterable[RawItem]:
        ...


__all__ = ["Cursor", "SourceAdapter", "RawItem"]
