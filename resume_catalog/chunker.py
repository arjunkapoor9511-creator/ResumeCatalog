"""Each extracted LineEntry is already a complete chunk (one bullet point
or one paragraph block), bullet/paragraph tag included -- extraction does
the real work; this just passes them through.
"""

from __future__ import annotations

from resume_catalog.extractors import LineEntry


def split_into_chunks(lines: list[LineEntry]) -> list[LineEntry]:
    return lines
