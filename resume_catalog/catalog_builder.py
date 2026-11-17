"""Dedup chunks by exact match after normalization, logging each chunk
as either a new catalog entry or a duplicate of an existing one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip()).lower()


@dataclass
class CatalogEntry:
    text: str
    source: str
    chunk_type: str
    seen_count: int = 1


class CatalogBuilder:
    def __init__(self):
        self._entries: dict[str, CatalogEntry] = {}
        self._order: list[str] = []

    def add(self, text: str, source: str, is_bullet: bool) -> None:
        key = normalize(text)
        if not key:
            return
        if key in self._entries:
            entry = self._entries[key]
            entry.seen_count += 1
            logger.debug("Duplicate (seen %dx): %r", entry.seen_count, text)
        else:
            chunk_type = "bullet" if is_bullet else "paragraph"
            self._entries[key] = CatalogEntry(text=text, source=Path(source).name, chunk_type=chunk_type)
            self._order.append(key)
            logger.info("New chunk: %r", text)

    def to_list(self) -> list[dict]:
        return [
            {
                "text": self._entries[key].text,
                "type": self._entries[key].chunk_type,
                "source": self._entries[key].source,
            }
            for key in self._order
        ]

    def __len__(self) -> int:
        return len(self._entries)
